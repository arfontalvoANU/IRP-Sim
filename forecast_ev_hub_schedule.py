"""Synthetic EPM EV-charging event forecasts for a fixed Ballarat truck fleet.

The fleet is always EV_001 ... EV_100, with one scheduled trip/hub visit per
active truck per day (100 weekdays, random 30 distinct trucks on weekends).
Arrival and departure deadlines are forecast independently of charging times.

Unlike the first version, each trip has a sampled distance and payload/driving
multiplier, a persistent truck-specific efficiency, individual trip duration,
refrigeration demand, and a calculated arrival SOC. Each EV's *planned* departure
SOC from the previous visit becomes its starting SOC at its next visit.
EPM energy is grid-side kWh (battery energy / charger efficiency).

All variability is synthetic and configurable, not an observed fleet forecast.
With --weather-csv, trip-period weather samples are used when available;
otherwise a supplied daily value or the default ambient temperature is used.

The detailed forecast also includes estimated FCFS charger connection/finish
for diagnostic purposes. The EPM may optimise charging within the fixed arrival
and planned departure window instead of using this FCFS charging schedule.
"""

# %% Imports and baseline parameters
import argparse
import heapq
from pathlib import Path
import numpy as np
import pandas as pd

TIMEZONE = "Australia/Melbourne"
FLEET_SIZE = 100
WEEKEND_ACTIVE_FRACTION = 0.30
FLEET_IDS = tuple(f"EV_{i:03d}" for i in range(1, FLEET_SIZE + 1))
BATTERY_KWH = 540.0
CHARGER_POWER_KW = 250.0
VEHICLE_POWER_KW = 250.0
CHARGER_EFFICIENCY = 0.95
TARGET_SOC = 0.90
TRIP_DISTANCE_KM = 230.0
TERRAIN_FACTOR = 1.05
BASE_KWH_PER_KM = 1.2
AVERAGE_SPEED_KMH = 80.0
TRU_BASE_KW = 8.0
TRU_TEMP_COEFFICIENT_KW_PER_C = 0.3
AUXILIARY_KW = 0.8
MIN_ARRIVAL_SOC = 0.10


def power_at_soc(peak_power_kw: float, soc: float) -> float:
    """Piecewise SOC power taper from the original EV charging notebook."""
    if soc <= 0.30:
        return peak_power_kw
    if soc <= 0.50:
        return peak_power_kw * (1.0 - 0.15 * (soc - 0.30) / 0.20)
    if soc <= 0.70:
        return peak_power_kw * (0.85 - 0.20 * (soc - 0.50) / 0.20)
    if soc <= 0.80:
        return peak_power_kw * (0.65 - 0.15 * (soc - 0.70) / 0.10)
    return peak_power_kw * (0.50 - 0.25 * (soc - 0.80) / 0.20)


def trip_energy_components_kwh(distance_km: float, consumption_kwh_km: float, duration_h: float, ambient_c: float, payload_factor: float = 1.0) -> dict:
    """Energy per trip: traction + refrigerated trailer + other auxiliaries."""
    traction = consumption_kwh_km * distance_km * TERRAIN_FACTOR * payload_factor
    if ambient_c < 5.0:
        traction *= 1.15
    elif ambient_c > 30.0:
        traction *= 1.05
    tru_kw = TRU_BASE_KW + TRU_TEMP_COEFFICIENT_KW_PER_C * max(0.0, ambient_c - 20.0)
    refrigeration = tru_kw * duration_h
    auxiliaries = AUXILIARY_KW * duration_h
    return {"traction": traction, "refrigeration": refrigeration, "auxiliaries": auxiliaries, "total": traction + refrigeration + auxiliaries}


def trip_energy_kwh(ambient_c: float) -> float:
    """Backward-compatible fixed-route baseline calculation (not the new forecast)."""
    return trip_energy_components_kwh(TRIP_DISTANCE_KM, BASE_KWH_PER_KM, TRIP_DISTANCE_KM / AVERAGE_SPEED_KMH, ambient_c)["total"]


def charge_duration_minutes(energy_to_grid_kwh: float, start_soc: float, step_minutes: float = 1.0) -> float:
    """Integrate actual requested grid kWh, with a fractional final time step."""
    if not 0 <= start_soc < 1 or energy_to_grid_kwh < 0 or step_minutes <= 0:
        raise ValueError("Invalid SOC, charging energy, or integration step")
    current_soc = start_soc
    grid_remaining = energy_to_grid_kwh
    minutes = 0.0
    while grid_remaining > 1e-8:
        power_kw = power_at_soc(min(CHARGER_POWER_KW, VEHICLE_POWER_KW), current_soc)
        if power_kw <= 0:
            raise ValueError("Invalid charging power")
        grid_step_kwh = min(grid_remaining, power_kw * step_minutes / 60.0)
        current_soc += (grid_step_kwh * CHARGER_EFFICIENCY) / BATTERY_KWH
        minutes += 60.0 * grid_step_kwh / power_kw
        grid_remaining -= grid_step_kwh
    return minutes


def local_wall_timestamp(day: pd.Timestamp, hour_of_day: float) -> pd.Timestamp:
    """Convert a local hour to a timezone-aware timestamp without DST index drift."""
    naive = pd.Timestamp(day.date()) + pd.Timedelta(hours=float(hour_of_day))
    return naive.tz_localize(TIMEZONE, ambiguous="raise", nonexistent="shift_forward")


def load_weather(weather_csv: str | None) -> tuple[pd.Series | None, pd.Series | None]:
    """Read optional datetime,temp_ambient CSV; return hourly data + daily means."""
    if weather_csv is None:
        return None, None
    data = pd.read_csv(weather_csv)
    if not {"datetime", "temp_ambient"}.issubset(data.columns):
        raise ValueError("Weather CSV must have 'datetime' and 'temp_ambient' columns")
    times = pd.to_datetime(data["datetime"], utc=True) if data["datetime"].astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).any() else pd.to_datetime(data["datetime"])
    if times.dt.tz is None:
        times = times.dt.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
    else:
        times = times.dt.tz_convert(TIMEZONE)
    series = pd.Series(pd.to_numeric(data["temp_ambient"], errors="coerce").values, index=pd.DatetimeIndex(times), name="temp_ambient").sort_index().dropna()
    if series.empty:
        raise ValueError("Weather CSV does not contain usable temperatures")
    daily = series.groupby(series.index.date).mean()
    return series, daily


def trip_ambient_c(departure: pd.Timestamp, arrival: pd.Timestamp, hourly_weather: pd.Series | None, daily_weather: pd.Series | None, fallback: float) -> float:
    """Mean available samples during trip, otherwise daily mean or fallback."""
    if hourly_weather is not None:
        # Captures both hourly forecasts (trip spans several samples) and
        # higher-resolution weather, without manufacturing hourly temperatures.
        samples = hourly_weather.loc[departure.floor("h"):arrival.floor("h")]
        if len(samples):
            return float(samples.mean())
    if daily_weather is not None and departure.date() in daily_weather.index:
        return float(daily_weather.loc[departure.date()])
    return float(fallback)



# %% FCFS charger allocation and energy-conserving hourly aggregation
def _fcfs_schedule(arrivals, departures, grid_kwh, arrival_socs, n_chargers: int, turnover_min: float = 5.0) -> list[dict]:
    """One FCFS allocator shared by initial forecasts and capacity scenarios."""
    if n_chargers < 1 or turnover_min < 0:
        raise ValueError("n_chargers must be positive and turnover_min must be nonnegative")
    if not arrivals:
        return []
    order = sorted(range(len(arrivals)), key=lambda i: pd.Timestamp(arrivals[i]))
    initial = pd.Timestamp(arrivals[order[0]]) - pd.Timedelta(days=1)
    heap = [(initial, charger) for charger in range(1, n_chargers + 1)]
    heapq.heapify(heap)
    assignments = [None] * len(arrivals)
    for i in order:
        arrival = pd.Timestamp(arrivals[i])
        deadline = pd.Timestamp(departures[i])
        available, charger = heapq.heappop(heap)
        connection = max(arrival, available)
        duration_min = charge_duration_minutes(float(grid_kwh[i]), float(arrival_socs[i]))
        finish = connection + pd.Timedelta(minutes=duration_min)
        heapq.heappush(heap, (finish + pd.Timedelta(minutes=turnover_min), charger))
        slack_min = (deadline - finish).total_seconds() / 60.0
        assignments[i] = {"Id Charging Point": f"CP_{charger:02d}", "Connection Time": connection, "Estimated Charge Finish": finish, "Queue Wait (min)": round(max(0.0, (connection - arrival).total_seconds() / 60.0), 2), "Charge Duration (min)": round(duration_min, 2), "Available Charging Slack (min)": round(slack_min, 2), "Deadline Feasible": bool(slack_min >= -1e-7)}
    return assignments


def reschedule_forecast(forecast: pd.DataFrame, n_chargers: int, turnover_min: float = 5.0) -> pd.DataFrame:
    """Reallocate a FIXED set of trips to another number of 250 kW CCS points.

    Arrival/departure windows, EV IDs, trip energy, and SOC remain unchanged.
    FCFS can miss the planned departure deadline; this is flagged, and a
    hypothetical missed charge does not feed back into next-day SOC.
    """
    fields = ["Arrival Time", "Departure Time", "Energy To Charge (kWh)", "Arrival SOC"]
    missing = [field for field in fields if field not in forecast]
    if missing:
        raise ValueError(f"Detailed forecast missing fields: {missing}")
    updated = forecast.copy()
    assignments = _fcfs_schedule(updated["Arrival Time"].tolist(), updated["Departure Time"].tolist(), updated["Energy To Charge (kWh)"].tolist(), updated["Arrival SOC"].tolist(), n_chargers, turnover_min)
    for column in ("Id Charging Point", "Connection Time", "Estimated Charge Finish", "Queue Wait (min)", "Charge Duration (min)", "Available Charging Slack (min)", "Deadline Feasible"):
        updated[column] = [entry[column] for entry in assignments]
    return updated


def forecast_to_hourly_load(events: pd.DataFrame, hours: pd.DatetimeIndex, step_minutes: float = 1.0) -> pd.DataFrame:
    """Convert FCFS charging sessions to average hourly grid power (kW and MW).

    Uses the SAME SOC taper and charger limits as charge_duration_minutes().
    All timestamps are compared in explicit nanoseconds to support pandas
    DatetimeIndex resolutions s/ms/us/ns and daylight-saving transitions.
    The last partial timestep is allocated proportionally, conserving each
    event's requested grid kWh exactly across hourly boundaries. Sessions
    outside the supplied hourly index raise instead of silently disappearing.
    """
    hours = pd.DatetimeIndex(hours)
    if hours.tz is None:
        raise ValueError("Hourly time index must be timezone-aware")
    if not 0 < step_minutes <= 60:
        raise ValueError("step_minutes must be in (0, 60]")
    required = {"Connection Time", "Arrival SOC", "Energy To Charge (kWh)"}
    if not required.issubset(events.columns):
        raise ValueError(f"Missing forecast columns: {sorted(required - set(events.columns))}")
    hour_ns = hours.as_unit("ns").asi8
    ns_per_hour = 3_600_000_000_000
    if len(hour_ns) == 0 or (len(hour_ns) > 1 and np.any(np.diff(hour_ns) != ns_per_hour)):
        raise ValueError("hours must be a continuous hourly time index (absolute elapsed hours)")
    energy_hour_kwh = np.zeros(len(hour_ns), dtype=float)
    n_sessions = np.zeros(len(hour_ns), dtype=int)
    n_sessions_touching = np.zeros(len(hour_ns), dtype=int)
    mean_active_chargers = np.zeros(len(hour_ns), dtype=float)
    hour_events = [[] for _ in range(len(hour_ns))]
    step_kwh_factor = float(step_minutes) / 60.0
    peak_kw = min(CHARGER_POWER_KW, VEHICLE_POWER_KW)
    lower_ns, upper_ns = int(hour_ns[0]), int(hour_ns[-1] + ns_per_hour)

    ix_start = events.columns.get_loc("Connection Time")
    ix_soc = events.columns.get_loc("Arrival SOC")
    ix_energy = events.columns.get_loc("Energy To Charge (kWh)")
    for row in events.itertuples(index=False, name=None):
        t_ns = int(pd.Timestamp(row[ix_start]).value)
        soc = float(row[ix_soc])
        remaining_kwh = float(row[ix_energy])
        if not (np.isfinite(soc) and 0 <= soc < 1 and np.isfinite(remaining_kwh) and remaining_kwh >= 0):
            raise ValueError("Charging event has an invalid SOC or charging energy")
        if remaining_kwh <= 0:
            continue
        if t_ns < lower_ns or t_ns >= upper_ns:
            raise ValueError(f"Charging connection {pd.Timestamp(row[ix_start])} falls outside the hourly index")
        start_ns = t_ns
        hour_pos = int(np.searchsorted(hour_ns, t_ns, side="right") - 1)
        touched = set()
        while remaining_kwh > 1e-9:
            power_kw = power_at_soc(peak_kw, soc)
            step_energy_kwh = min(remaining_kwh, power_kw * step_kwh_factor)
            dt_ns = max(1, round(step_energy_kwh * ns_per_hour / power_kw))
            next_ns = t_ns + dt_ns
            if next_ns > upper_ns:
                raise ValueError("Charging session extends beyond hourly index; extend hours to include all charging")
            piece_start_ns = t_ns
            while piece_start_ns < next_ns:
                if hour_pos >= len(hour_ns):
                    raise ValueError("Charging session extends beyond hourly index")
                piece_end_ns = min(next_ns, int(hour_ns[hour_pos] + ns_per_hour))
                energy_hour_kwh[hour_pos] += step_energy_kwh * (piece_end_ns - piece_start_ns) / dt_ns
                touched.add(hour_pos)
                piece_start_ns = piece_end_ns
                if piece_start_ns == hour_ns[hour_pos] + ns_per_hour:
                    hour_pos += 1
            soc += step_energy_kwh * CHARGER_EFFICIENCY / BATTERY_KWH
            remaining_kwh -= step_energy_kwh
            t_ns = next_ns
        # A session can touch an hour even if it is not concurrent with other
        # sessions touching that hour. Compute true instantaneous peak occupancy.
        session_pos = int(np.searchsorted(hour_ns, start_ns, side="right") - 1)
        while session_pos < len(hour_ns) and hour_ns[session_pos] < t_ns:
            piece_a = max(start_ns, int(hour_ns[session_pos]))
            piece_b = min(t_ns, int(hour_ns[session_pos] + ns_per_hour))
            if piece_b > piece_a:
                n_sessions_touching[session_pos] += 1
                mean_active_chargers[session_pos] += (piece_b - piece_a) / ns_per_hour
                hour_events[session_pos].append((piece_a, 1))
                hour_events[session_pos].append((piece_b, -1))
            session_pos += 1

    for hour_idx, edges in enumerate(hour_events):
        if not edges:
            continue
        occupancy = peak_occupancy = 0
        for _, change in sorted(edges, key=lambda x: (x[0], x[1])):
            occupancy += change
            peak_occupancy = max(peak_occupancy, occupancy)
        n_sessions[hour_idx] = peak_occupancy
    return pd.DataFrame({"n_sessions": n_sessions, "mean_active_chargers": mean_active_chargers, "n_sessions_touching": n_sessions_touching, "load_kw": energy_hour_kwh, "load_mw": energy_hour_kwh / 1000.0}, index=hours)


# %% Vehicle-level, day-level and charging-point event forecast
def build_forecast(start_date: str, end_date: str, n_chargers: int = 42, ambient_c: float = 20.0, seed: int = 42, dwell_mean_h: float = 4.0, dwell_sd_h: float = 0.75, weather_csv: str | None = None, turnover_min: float = 5.0, trip_distance_mean_km: float = 230.0, trip_distance_sd_km: float = 30.0, vehicle_consumption_sd: float = 0.07, payload_factor_sd: float = 0.06, target_soc_sd: float = 0.025, initial_soc: float = 0.90) -> pd.DataFrame:
    """Create an EPM visit forecast while retaining fixed EV_001 ... EV_100 IDs.

    Each EV has a persistent underlying efficiency and preferred charging SOC.
    Route distance, payload/driving factor, temperature, and requested target
    SOC may vary by visit. SOC history carries forward the *planned* target SOC
    under the modelling assumption that requested charges complete. If a
    scheduling deadline cannot be met, it is flagged (not silently moved).
    Infeasible randomly sampled trips are resampled to maintain >=10% arrival
    SOC instead of recording impossible negative battery energy.
    """
    if n_chargers <= 0 or turnover_min < 0:
        raise ValueError("n_chargers must be positive and turnover_min nonnegative")
    if dwell_mean_h <= 0 or dwell_sd_h < 0 or trip_distance_mean_km <= 0 or trip_distance_sd_km < 0 or vehicle_consumption_sd < 0 or payload_factor_sd < 0 or target_soc_sd < 0:
        raise ValueError("Invalid trip or dwell distribution parameters")
    if not MIN_ARRIVAL_SOC < initial_soc <= 0.95:
        raise ValueError("initial_soc must exceed minimum arrival SOC and be <=0.95")
    days = pd.date_range(start_date, end_date, freq="D", tz=TIMEZONE)
    if days.empty:
        raise ValueError("End date must not precede start date")

    # Separate RNGs prevent random route/energy variation from inadvertently
    # changing the original vehicle selection and arrival-time random sequence.
    arrival_rng = np.random.default_rng(seed)
    fleet_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    energy_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    hourly_weather, daily_weather = load_weather(weather_csv)
    fleet_consumption = {ev: float(np.clip(energy_rng.normal(BASE_KWH_PER_KM, vehicle_consumption_sd), 1.05, 1.35)) for ev in FLEET_IDS}
    fleet_target = {ev: float(np.clip(energy_rng.normal(TARGET_SOC, target_soc_sd), 0.85, 0.95)) for ev in FLEET_IDS}
    planned_soc = {ev: float(initial_soc) for ev in FLEET_IDS}
    candidates = []

    # Stage A: operational trips and hub availability windows.
    for day in days:
        n_trucks = int(FLEET_SIZE * WEEKEND_ACTIVE_FRACTION) if day.dayofweek >= 5 else FLEET_SIZE
        active_ids = fleet_rng.choice(FLEET_IDS, size=n_trucks, replace=False)
        route_depart_h = 5.0 + 9.0 * arrival_rng.beta(2.0, 5.0, n_trucks)
        original_duration_h = np.clip(arrival_rng.normal(3.0, 0.5, n_trucks), 2.1, 4.5)
        dwell_h = np.clip(arrival_rng.normal(dwell_mean_h, dwell_sd_h, n_trucks), 3.0, 6.5)
        for ix, ev in enumerate(active_ids):
            ev = str(ev)
            prior_departure_soc = planned_soc[ev]
            target_soc = float(np.clip(fleet_target[ev] + energy_rng.normal(0, 0.012), 0.85, 0.95))
            # Resample only physically infeasible daily route assignments.
            for attempts in range(100):
                route_km = float(np.clip(energy_rng.normal(trip_distance_mean_km, trip_distance_sd_km), 150.0, 270.0))
                payload_factor = float(np.clip(energy_rng.normal(1.0, payload_factor_sd), 0.85, 1.15))
                trip_h = float(np.clip(original_duration_h[ix] * route_km / TRIP_DISTANCE_KM, 1.8, 5.0))
                route_departure = local_wall_timestamp(day, route_depart_h[ix])
                hub_arrival = route_departure + pd.Timedelta(hours=trip_h)
                temp = trip_ambient_c(route_departure, hub_arrival, hourly_weather, daily_weather, ambient_c)
                components = trip_energy_components_kwh(route_km, fleet_consumption[ev], trip_h, temp, payload_factor)
                arrival_soc = prior_departure_soc - components["total"] / BATTERY_KWH
                if arrival_soc >= MIN_ARRIVAL_SOC:
                    break
            else:
                raise ValueError(f"Cannot generate a feasible trip for {ev} on {day.date()}; try shorter routes or a higher starting SOC")
            # State-of-charge accounting: E_batt=(SOC_target-SOC_arrival)*capacity;
            # E_grid=E_batt/charger_efficiency. The initial departure SOC is
            # 90%, but subsequent daily starting SOC follows the prior visit.
            if target_soc <= arrival_soc:
                target_soc = min(0.95, arrival_soc + 0.02)
            battery_requested = BATTERY_KWH * (target_soc - arrival_soc)
            grid_requested = battery_requested / CHARGER_EFFICIENCY
            hub_departure = hub_arrival + pd.Timedelta(hours=float(dwell_h[ix]))
            candidates.append({"vehicle_id": ev, "arrival": hub_arrival, "departure": hub_departure, "dwell_h": float(dwell_h[ix]), "route_departure": route_departure, "trip_duration_h": trip_h, "distance_km": route_km, "vehicle_base_kwh_per_km": fleet_consumption[ev], "payload_factor": payload_factor, "temp_c": temp, "traction_kwh": components["traction"], "refrigeration_kwh": components["refrigeration"], "auxiliary_kwh": components["auxiliaries"], "trip_energy_kwh": components["total"], "battery_energy_kwh": battery_requested, "grid_energy_kwh": grid_requested, "departure_soc_before_trip": prior_departure_soc, "arrival_soc": arrival_soc, "target_soc": target_soc, "trip_resampling_attempts": attempts})
            planned_soc[ev] = target_soc

    # Stage B: FCFS charger assignment, independent of planned departure window.
    candidates.sort(key=lambda x: x["arrival"])
    assignments = _fcfs_schedule([e["arrival"] for e in candidates], [e["departure"] for e in candidates], [e["grid_energy_kwh"] for e in candidates], [e["arrival_soc"] for e in candidates], n_chargers, turnover_min)
    records = []
    for entry, assigned in zip(candidates, assignments):
        records.append({
            "Id EV": entry["vehicle_id"],
            "Arrival Time": entry["arrival"],
            "Departure Time": entry["departure"],
            "Energy To Charge (kWh)": round(entry["grid_energy_kwh"], 2),
            "Id Charging Point": assigned["Id Charging Point"],
            "Route Departure Time": entry["route_departure"],
            "Trip Distance (km)": round(entry["distance_km"], 2),
            "Trip Duration (h)": round(entry["trip_duration_h"], 3),
            "Vehicle Base Consumption (kWh/km)": round(entry["vehicle_base_kwh_per_km"], 4),
            "Daily Payload/Driving Factor": round(entry["payload_factor"], 4),
            "Ambient (C)": round(entry["temp_c"], 2),
            "Traction Energy (kWh)": round(entry["traction_kwh"], 2),
            "Refrigeration Energy (kWh)": round(entry["refrigeration_kwh"], 2),
            "Auxiliary Energy (kWh)": round(entry["auxiliary_kwh"], 2),
            "Trip Energy (kWh)": round(entry["trip_energy_kwh"], 2),
            "Battery Energy Requested (kWh)": round(entry["battery_energy_kwh"], 2),
            "Previous Departure SOC": round(entry["departure_soc_before_trip"], 5),
            "Arrival SOC": round(entry["arrival_soc"], 5),
            "Target SOC": round(entry["target_soc"], 5),
            "Connection Time": assigned["Connection Time"],
            "Estimated Charge Finish": assigned["Estimated Charge Finish"],
            "Queue Wait (min)": assigned["Queue Wait (min)"],
            "Planned Dwell (h)": round(entry["dwell_h"], 3),
            "Charge Duration (min)": assigned["Charge Duration (min)"],
            "Available Charging Slack (min)": assigned["Available Charging Slack (min)"],
            "Deadline Feasible": assigned["Deadline Feasible"],
            "Trip Resampling Attempts": entry["trip_resampling_attempts"],
        })
    return pd.DataFrame(records)


def save_forecast(forecast: pd.DataFrame, out_prefix: str) -> None:
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    epm_cols = ["Id EV", "Arrival Time", "Departure Time", "Energy To Charge (kWh)", "Id Charging Point"]
    forecast[epm_cols].to_csv(f"{prefix}_EPM.csv", index=False, date_format="%Y-%m-%d %H:%M:%S%z")
    forecast.to_csv(f"{prefix}_detailed.csv", index=False, date_format="%Y-%m-%d %H:%M:%S%z")
    if len(forecast):
        print(f"Hub visits: {len(forecast):,}")
        print(f"Distinct fleet IDs: {forecast['Id EV'].nunique()} / {FLEET_SIZE}")
        print(f"Charger points used: {forecast['Id Charging Point'].nunique()}")
        print(f"Energy to charge: min {forecast['Energy To Charge (kWh)'].min():.1f}, mean {forecast['Energy To Charge (kWh)'].mean():.1f}, max {forecast['Energy To Charge (kWh)'].max():.1f} grid kWh")
        print(f"Total requested grid energy: {forecast['Energy To Charge (kWh)'].sum()/1000:.2f} MWh")
        print(f"Mean planned dwell: {forecast['Planned Dwell (h)'].mean():.2f} h")
        print(f"Mean queue wait: {forecast['Queue Wait (min)'].mean():.1f} min")
        print(f"Infeasible FCFS charging deadlines: {(~forecast['Deadline Feasible']).sum():,}")
        print(f"Output: {prefix}_EPM.csv")
        print(f"Output: {prefix}_detailed.csv")
        print(forecast[epm_cols].head(6).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fixed-fleet EPM charging-event forecasts with individual truck energy requirements")
    parser.add_argument("--start-date", default="2026-11-02", help="First arrival date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-11-08", help="Last arrival date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--chargers", type=int, default=42, help="Number of CCS 250 kW charging points")
    parser.add_argument("--temperature-c", type=float, default=20.0, help="Constant weather scenario temperature")
    parser.add_argument("--weather-csv", default=None, help="Optional CSV with datetime,temp_ambient (hourly or daily)")
    parser.add_argument("--dwell-hours", type=float, default=4.0, help="Mean planned hub dwell (not charge duration)")
    parser.add_argument("--dwell-std-hours", type=float, default=0.75, help="Planned dwell standard deviation")
    parser.add_argument("--trip-distance-mean-km", type=float, default=230.0)
    parser.add_argument("--trip-distance-sd-km", type=float, default=30.0)
    parser.add_argument("--vehicle-consumption-sd", type=float, default=0.07, help="Across-fleet baseline kWh/km standard deviation")
    parser.add_argument("--payload-factor-sd", type=float, default=0.06, help="Daily traction multiplier standard deviation")
    parser.add_argument("--target-soc-sd", type=float, default=0.025, help="Across-fleet preferred target SOC standard deviation")
    parser.add_argument("--initial-soc", type=float, default=0.90, help="Starting SOC at first simulated trip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="output/ev_charging/ballarat_hub_forecast")
    args = parser.parse_args()
    result = build_forecast(args.start_date, args.end_date, n_chargers=args.chargers, ambient_c=args.temperature_c, seed=args.seed, dwell_mean_h=args.dwell_hours, dwell_sd_h=args.dwell_std_hours, weather_csv=args.weather_csv, trip_distance_mean_km=args.trip_distance_mean_km, trip_distance_sd_km=args.trip_distance_sd_km, vehicle_consumption_sd=args.vehicle_consumption_sd, payload_factor_sd=args.payload_factor_sd, target_soc_sd=args.target_soc_sd, initial_soc=args.initial_soc)
    save_forecast(result, args.out_prefix)

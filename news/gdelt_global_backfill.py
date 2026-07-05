#!/usr/bin/env python3
import io, zipfile, requests, sys
from datetime import date, timedelta, datetime
from sqlalchemy import create_engine, text

sys.path.insert(0, "/opt/Franco-Investment-Universe")
from assets.const import DB_PARAMS

DDL = """
CREATE TABLE IF NOT EXISTS news_event_daily_global (
  event_date DATE PRIMARY KEY,
  event_count BIGINT NOT NULL DEFAULT 0,
  avg_tone DOUBLE PRECISION,
  avg_goldstein DOUBLE PRECISION,
  conflict_events BIGINT NOT NULL DEFAULT 0,
  war_events BIGINT NOT NULL DEFAULT 0,
  sanctions_events BIGINT NOT NULL DEFAULT 0,
  shipping_events BIGINT NOT NULL DEFAULT 0,
  sa_related_events BIGINT NOT NULL DEFAULT 0,
  major_power_tension_events BIGINT NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
"""

UP = text("""
INSERT INTO news_event_daily_global (
  event_date,event_count,avg_tone,avg_goldstein,conflict_events,war_events,sanctions_events,shipping_events,sa_related_events,major_power_tension_events,updated_at
) VALUES (
  :event_date,:event_count,:avg_tone,:avg_goldstein,:conflict_events,:war_events,:sanctions_events,:shipping_events,:sa_related_events,:major_power_tension_events,NOW()
)
ON CONFLICT (event_date) DO UPDATE SET
  event_count=EXCLUDED.event_count,
  avg_tone=EXCLUDED.avg_tone,
  avg_goldstein=EXCLUDED.avg_goldstein,
  conflict_events=EXCLUDED.conflict_events,
  war_events=EXCLUDED.war_events,
  sanctions_events=EXCLUDED.sanctions_events,
  shipping_events=EXCLUDED.shipping_events,
  sa_related_events=EXCLUDED.sa_related_events,
  major_power_tension_events=EXCLUDED.major_power_tension_events,
  updated_at=NOW();
""")


def f(x, d=0.0):
    try:
        return float(x)
    except:
        return d


def i(x, d=0):
    try:
        return int(float(x))
    except:
        return d


def process_day(day: date):
    url = f"http://data.gdeltproject.org/events/{day.strftime('%Y%m%d')}.export.CSV.zip"
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        return None

    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    lines = z.read(name).decode("latin-1", errors="ignore").splitlines()

    n = 0
    tone_sum = 0.0
    gold_sum = 0.0
    conflict = war = sanctions = shipping = sa_rel = tension = 0

    for line in lines:
        c = line.split("\t")
        if len(c) < 35:
            continue
        n += 1
        event_code = c[26] if len(c) > 26 else ""
        root_code = c[28] if len(c) > 28 else ""
        actor1_cc = c[7] if len(c) > 7 else ""
        actor2_cc = c[17] if len(c) > 17 else ""
        tone = f(c[34], 0.0)
        gold = f(c[30], 0.0)
        url = c[-1].lower() if c else ""

        tone_sum += tone
        gold_sum += gold

        # CAMEO 19/20 are fight/violence umbrellas, 190+ war-related
        if (
            root_code in {"19", "20"}
            or event_code.startswith("19")
            or event_code.startswith("20")
        ):
            conflict += 1
        if (
            event_code.startswith("190")
            or event_code.startswith("191")
            or event_code.startswith("192")
            or event_code.startswith("193")
        ):
            war += 1
        if "sanction" in url or event_code.startswith("1122"):
            sanctions += 1
        if any(
            k in url
            for k in [
                "shipping",
                "freight",
                "container",
                "red-sea",
                "suez",
                "strait",
                "port disruption",
            ]
        ):
            shipping += 1

        if actor1_cc == "SAF" or actor2_cc == "SAF" or ".za/" in url:
            sa_rel += 1

        # rough major-power tension proxy
        cc_pair = {actor1_cc, actor2_cc}
        if (
            ("USA" in cc_pair and "CHN" in cc_pair)
            or ("RUS" in cc_pair and "USA" in cc_pair)
            or ("RUS" in cc_pair and "UKR" in cc_pair)
        ):
            tension += 1

    if n == 0:
        return None

    return {
        "event_date": day,
        "event_count": n,
        "avg_tone": tone_sum / n,
        "avg_goldstein": gold_sum / n,
        "conflict_events": conflict,
        "war_events": war,
        "sanctions_events": sanctions,
        "shipping_events": shipping,
        "sa_related_events": sa_rel,
        "major_power_tension_events": tension,
    }


def main(start_s: str, end_s: str):
    eng = create_engine(
        f"postgresql://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    )
    with eng.begin() as c:
        c.execute(text(DDL))

    start = datetime.strptime(start_s, "%Y-%m-%d").date()
    end = datetime.strptime(end_s, "%Y-%m-%d").date()

    d = start
    done = 0
    while d <= end:
        try:
            row = process_day(d)
            if row:
                with eng.begin() as c:
                    c.execute(UP, row)
                done += 1
                print(f"{d} upserted events={row['event_count']}")
            else:
                print(f"{d} missing/empty")
        except Exception as e:
            print(f"{d} error {e}")
        d += timedelta(days=1)

    print(f"DONE_DAYS={done}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        end = date.today()
        start = end - timedelta(days=365 * 10)
        main(start.isoformat(), end.isoformat())
    else:
        main(sys.argv[1], sys.argv[2])

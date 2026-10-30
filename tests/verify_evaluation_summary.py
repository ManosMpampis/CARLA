"""Machine aggregation keeps cases separate and excludes failed measurements."""
import csv
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.evaluation_summary import write_evaluation_overview


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for machine, speed in (("machine-1-1.txt", 100), ("machine-1-2.txt", 200)):
            directory = root / machine
            directory.mkdir()
            rows = []
            for case in ("input_512", "input_128"):
                rows.append(dict(job=f"ae/default/selected/{case}", experiment_key="ae/default",
                                 checkpoint_name="selected", case=case, fname=machine,
                                 status="failed" if case == "input_128" and speed == 200 else "completed",
                                 evaluation_settings={"input_window": int(case.split('_')[1])},
                                 inference_speed={"windows_per_second": speed, "ms_per_window": 1000 / speed},
                                 evaluation={"timeseries": {"calibrated": {"f1_no_pa": speed / 1000},
                                                             "vus_pr": 0.5, "vus_roc": 0.8}}))
            (directory / "summary.json").write_text(json.dumps(rows))
        write_evaluation_overview(root)
        with (root / "summary.csv").open() as stream:
            assert len(list(csv.DictReader(stream))) == 4
        with (root / "summary_aggregated.csv").open() as stream:
            aggregated = {row['case']: row for row in csv.DictReader(stream)}
        assert float(aggregated['input_512']['inference_windows_per_second_mean']) == 150
        assert int(aggregated['input_512']['machines_completed']) == 2
        assert float(aggregated['input_128']['inference_windows_per_second_mean']) == 100
        assert int(aggregated['input_128']['machines_completed']) == 1
        for case in aggregated:
            assert (root / 'cases' / case / 'summary.csv').is_file()
            assert (root / 'cases' / case / 'summary_aggregated.csv').is_file()
    print('Evaluation summaries: machine weights, separate cases, timing and failures OK')


if __name__ == '__main__':
    main()

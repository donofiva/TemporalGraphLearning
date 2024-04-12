import pandas as pd

from typing import Set
from src.abstraction.entities.Turbine import Turbine
from src.abstraction.entities.Day import Day
from src.abstraction.entities.Timeslot import Timeslot
from src.abstraction.entities.Dimension import Dimension
from src.reporting.Report import Report
from src.io.FileManager import FileManager
from src.io.File import File
from src.io.Directory import Directory


class ReportingManager:

    _report_to_file = {
        Report.LOCATIONS: File.LOCATIONS,
        Report.TIMESERIES: File.TIMESERIES
    }

    @staticmethod
    def _store_report(report_identifier: Report, report_data: pd.DataFrame):
        report_data.to_csv(
            path_or_buf=FileManager.resolve_file(
                directory=FileManager.resolve_directory(Directory.OUTPUTS),
                file=ReportingManager._report_to_file[report_identifier]
            ),
            index=False
        )

    @staticmethod
    def store_locations_report(turbines: Set[Turbine]):
        ReportingManager._store_report(
            report_identifier=Report.LOCATIONS,
            report_data=pd.DataFrame(
                [(turbine.index, turbine.x_axis, turbine.y_axis) for turbine in turbines],
                columns=['TURBINE', 'X_AXIS', 'Y_AXIS']
            )
        )

    @staticmethod
    def store_timeseries_report(turbines: Set[Turbine]):

        # Generate report
        report = pd.DataFrame(
            [
                (
                    turbine.index,
                    day.index,
                    str(timeslot),
                    *[
                        turbine.get_timeseries_by_dimension(dimension).get_datapoint_by_day_and_timeslot(
                            day=day,
                            timeslot=timeslot
                        )
                        for dimension in Dimension
                    ]
                )
                for turbine in turbines
                for day in Day.get_all()
                for timeslot in Timeslot.get_all()
            ],
            columns=[
                'TURBINE',
                'DAY',
                'TIMESLOT',
                *[dimension.name for dimension in Dimension]
            ]
        )

        ReportingManager._store_report(
            report_identifier=Report.TIMESERIES,
            report_data=report.sort_values(['TURBINE', 'DAY', 'TIMESLOT'])
        )


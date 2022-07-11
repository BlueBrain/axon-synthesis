"""Store the general configuration values."""
import luigi
import luigi_tools.parameter
import luigi_tools.task


class Config(luigi.task.Config):
    output_dir = luigi_tools.parameter.OptionalPathParameter(
        description="The directory in which all the results will be exported",
        default=None,
    )
    input_data_type = luigi.ChoiceParameter(
        description=("The type of input data to use."),
        choices=["biological_morphologies", "white_matter"],
        default="biological_morphologies",
    )
    atlas_path = luigi_tools.parameter.OptionalPathParameter(
        description="Atlas path", exists=True, default=None
    )
    atlas_region_filename = luigi.Parameter(
        description="Atlas regions file.",
        default="brain_regions",
    )
    atlas_flatmap_filename = luigi.OptionalParameter(
        description="Atlas flatmap file.",
        default=None,
    )
    atlas_hierarchy_filename = luigi.Parameter(
        description="Atlas hierarchy file.",
        default="hierarchy.json",
    )
    white_matter_file = luigi_tools.parameter.OptionalPathParameter(
        description="White matter file", exists=True, default=None
    )

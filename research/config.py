"""Store the general configuration values."""
import luigi
import luigi_tools


class Config(luigi.task.Config):
    atlas_path = luigi_tools.parameter.PathParameter(
        description="Atlas path", exists=True
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
    white_matter_file = luigi_tools.parameter.PathParameter(
        description="White matter file", exists=True
    )
    input_data_type = luigi.ChoiceParameter(
        description=(
            "The type of input data to use."
        ),
        choices=["biological_morphologies", "white_matter"],
        default="biological_morphologies",
    )

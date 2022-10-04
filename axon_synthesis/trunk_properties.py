"""Compute the long-range trunk properties after clustering."""
import json
import logging

import luigi
import luigi_tools
import numpy as np
import pandas as pd
from data_validation_framework.target import TaggedOutputLocalTarget
from neurom import load_morphology
from neurom.apps import morph_stats

from axon_synthesis.PCSF.clustering import ClusterTerminals
from axon_synthesis.utils import get_axons

logger = logging.getLogger(__name__)


class TrunkPropertiesOutputLocalTarget(TaggedOutputLocalTarget):
    """Target for clustering outputs."""

    __prefix = "trunk_properties"  # pylint: disable=unused-private-member


class LongRangeTrunkProperties(luigi_tools.task.WorkflowTask):
    """Task to cluster the terminals."""

    morphology_paths = luigi.OptionalPathParameter(
        description="Path to the CSV file containing the paths to the trunk morphologies.",
        default=None,
        exists=True,
    )

    def requires(self):
        return ClusterTerminals()

    def run(self):
        # Load morph paths
        logger.info("Extract trunk properties")
        morphology_paths = pd.read_csv(
            self.morphology_paths or self.input()["trunk_morphology_paths"].path,
            dtype={"morph_file": str},
        )
        long_range_trunk_props = []

        for idx, row in morphology_paths.iterrows():
            logger.info("Extract trunk properties from %s", row["morph_path"])
            trunk_morph = load_morphology(row["morph_path"])

            # Compute long-range trunk features that will be used for smoothing and jittering
            long_range_trunks = get_axons(trunk_morph)
            for num, axon in enumerate(long_range_trunks):
                trunk_stats = morph_stats.extract_stats(
                    axon,
                    {
                        "neurite": {
                            "segment_lengths": {"modes": ["raw", "mean", "std"]},
                            "segment_meander_angles": {"modes": ["raw", "mean", "std"]},
                        }
                    },
                )["axon"]
                long_range_trunk_props.append(
                    (
                        row["morph_file"],
                        num,
                        json.dumps(np.array(trunk_stats["raw_segment_lengths"]).tolist()),
                        trunk_stats["mean_segment_lengths"],
                        trunk_stats["std_segment_lengths"],
                        json.dumps(np.array(trunk_stats["raw_segment_meander_angles"]).tolist()),
                        trunk_stats["mean_segment_meander_angles"],
                        trunk_stats["std_segment_meander_angles"],
                    )
                )

        # Export long-range trunk properties
        trunk_props_df = pd.DataFrame(
            long_range_trunk_props,
            columns=[
                "morph_file",
                "axon_id",
                "raw_segment_lengths",
                "mean_segment_lengths",
                "std_segment_lengths",
                "raw_segment_meander_angles",
                "mean_segment_meander_angles",
                "std_segment_meander_angles",
            ],
        )
        trunk_props_df.sort_values(["morph_file", "axon_id"], inplace=True)
        trunk_props_df.to_csv(self.output().path, index=False)
        logger.info("Exported trunk properties to %s", self.output().path)

    def output(self):
        return TrunkPropertiesOutputLocalTarget("trunk_properties.csv", create_parent=True)

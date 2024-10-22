# Changelog

## [0.1.5](https://github.com/BlueBrain/axon-synthesis/compare/0.1.4..0.1.5)

> 22 October 2024

### Documentation Changes

- Fix RTD (Adrien Berchet - [#4](https://github.com/BlueBrain/axon-synthesis/pull/4))

## [0.1.4](https://github.com/BlueBrain/axon-synthesis/compare/0.1.3..0.1.4)

> 22 October 2024

### Chores And Housekeeping

- Move repo URL (Adrien Berchet - [#2](https://github.com/BlueBrain/axon-synthesis/pull/2))
- Migrate the code to GitHub and open source it (Adrien Berchet - [#1](https://github.com/BlueBrain/axon-synthesis/pull/1))

## [0.1.3](https://github.com/BlueBrain/axon-synthesis/compare/0.1.2..0.1.3)

> 22 October 2024

### New Features

- Can pass a parameter file for tuft synthesis in mimic workflow (Adrien Berchet - [9b7aff7](https://github.com/BlueBrain/axon-synthesis/commit/9b7aff799e3f054f3b8c5e2af4caea9cf01f9eb6))
- Compute the maximum path extent in tuft properties (Adrien Berchet - [ff60d9e](https://github.com/BlueBrain/axon-synthesis/commit/ff60d9ef825e80d16011cb34c1526005f54f5047))

### Chores And Housekeeping

- Cast paths to str when getting morphology paths (Adrien Berchet - [0e94dd1](https://github.com/BlueBrain/axon-synthesis/commit/0e94dd14bb9be711c3ca93f8d88c05cb3927519a))

## [0.1.2](https://github.com/BlueBrain/axon-synthesis/compare/0.1.1..0.1.2)

> 22 October 2024

### Fixes

- Ensure morphology edges are optional (Adrien Berchet - [0ce1694](https://github.com/BlueBrain/axon-synthesis/commit/0ce169408f1d9ee240dc2df89fa92ee7716f514d))

### Chores And Housekeeping

- Use grower context to pass the boundary to the tuft grower (Adrien Berchet - [e5bd5ff](https://github.com/BlueBrain/axon-synthesis/commit/e5bd5ff20d69dcf534c8fda41100f6d65a33ea8b))

## [0.1.1](https://github.com/BlueBrain/axon-synthesis/compare/0.1.0..0.1.1)

> 22 October 2024

### Fixes

- Fix atlas CLI and remove useless warnings (Adrien Berchet - [c1e6afb](https://github.com/BlueBrain/axon-synthesis/commit/c1e6afb2d26a13db9a82ad2290df43e71ca9092b))

## 0.1.0

> 22 October 2024

### Build

- Bump NeuroTS and update the tests (Adrien Berchet - [956e284](https://github.com/BlueBrain/axon-synthesis/commit/956e284c0c3c25bd8d1823dfa3316b26b56fab95))
- Fix m2r2 dependency to docutils&lt;0.21 (Adrien Berchet - [80723f4](https://github.com/BlueBrain/axon-synthesis/commit/80723f4022899352fb17582e55a36e7f533571ef))

### New Features

- Support nested folders in input morphology directory (Adrien Berchet - [af71fac](https://github.com/BlueBrain/axon-synthesis/commit/af71facd35b0a41c5fbf52d9fae0ec9b18754c84))
- Add dummy preferred region workflow in mimic validation (Adrien Berchet - [7a0b876](https://github.com/BlueBrain/axon-synthesis/commit/7a0b876a6ca88fc592b40a1655712b2326da395b))
- Add tuft creation and grafting (Adrien Berchet - [f4dafe3](https://github.com/BlueBrain/axon-synthesis/commit/f4dafe3e6fef47f5bf1d337f333a32485737337f))
- Compute layer profile coefficients for sub regions (Adrien Berchet - [97625ff](https://github.com/BlueBrain/axon-synthesis/commit/97625ffa1c3930476d71bfd3f826155bac5f7e70))
- Add parallel computation (Adrien Berchet - [c6ab0c0](https://github.com/BlueBrain/axon-synthesis/commit/c6ab0c0f8546c24e24dc530159f3985cbdf1417e))
- Add trunk post-processing (Adrien Berchet - [5db264d](https://github.com/BlueBrain/axon-synthesis/commit/5db264d2403e9aef3cb5fd9e222f4c1cc040e4c2))
- Add smoothing and jittering for long-range trunk and some statistics on tuft distributions (Adrien Berchet - [9be74a1](https://github.com/BlueBrain/axon-synthesis/commit/9be74a161a335909b50fffb260b75d7228f56be9))
- Improve plots (Adrien Berchet - [fe65578](https://github.com/BlueBrain/axon-synthesis/commit/fe6557831aa61622be47c1853d0610c99b16f0f1))
- Add many options in CLI (Adrien Berchet - [264ac8a](https://github.com/BlueBrain/axon-synthesis/commit/264ac8a8f6cbf5e26304e4ebc27fa4484bba0135))
- Add functions to generate score matrices (Adrien Berchet - [bd495f8](https://github.com/BlueBrain/axon-synthesis/commit/bd495f82a5ceaa47feb5ad391b682522c8823fe0))
- Add favored regions and depth penalties to build the graph (Adrien Berchet - [a562aea](https://github.com/BlueBrain/axon-synthesis/commit/a562aea4e6e63335ffae41f4e52f8d1c642ddc08))
- Compute inputs in parallel (Adrien Berchet - [16f0336](https://github.com/BlueBrain/axon-synthesis/commit/16f0336cb724c97b80cdc3955e1aae15bdd4ef9f))
- Can pass multiple clustering configurations (Adrien Berchet - [a6434e4](https://github.com/BlueBrain/axon-synthesis/commit/a6434e420415766d7f25ded60e709c98a0eb02df))
- Add script to init empty morphs, add ST solution figure, add some config params (Adrien Berchet - [2720dec](https://github.com/BlueBrain/axon-synthesis/commit/2720decfcac09c4f69568bfe0a0141f94669de50))
- Add boundary handling in tuft synthesis (Adrien Berchet - [de004db](https://github.com/BlueBrain/axon-synthesis/commit/de004dbb3aaafc54fb6020a368b32e2f9d29fde4))
- Add example and handle multiple axons properly for input creation (Adrien Berchet - [d753d96](https://github.com/BlueBrain/axon-synthesis/commit/d753d964c8c7dbe2ec3c42ced48f96db6a36af74))
- Enable parallel computation for all mimic workflows (Adrien Berchet - [f407f02](https://github.com/BlueBrain/axon-synthesis/commit/f407f02e9b08b0a15956b74693705127ce05dc51))
- Improve random walk along the Steiner solution (Adrien Berchet - [29e3b7b](https://github.com/BlueBrain/axon-synthesis/commit/29e3b7b3083abe76f0b96d362886179c02c04c17))
- Parallelize terminal extraction and reduce logger entries from distributed scheduler (Adrien Berchet - [031eb77](https://github.com/BlueBrain/axon-synthesis/commit/031eb77e5a686f247c5a0b36cc26327833b0a228))
- Add projection intensities computation (Adrien Berchet - [6ff7da1](https://github.com/BlueBrain/axon-synthesis/commit/6ff7da11df3e5b6b6d42888ecc77b89a40497300))
- Add random walk along the Steiner solution (Adrien Berchet - [10ef6cb](https://github.com/BlueBrain/axon-synthesis/commit/10ef6cbac846b43ad8d57ddc01d5c473445c8d0a))
- Add parameters to use custom file locations instead of only from inputs directory (Adrien Berchet - [7e51a38](https://github.com/BlueBrain/axon-synthesis/commit/7e51a381f10c587c79c5b0f9eda526a7a55b460f))
- Improve projection intensities and add figures (Adrien Berchet - [9d002ab](https://github.com/BlueBrain/axon-synthesis/commit/9d002ab4cbda708709cefcad106f6af8eac953cf))
- Set trunk diameter to 1 by default (Adrien Berchet - [9ae19c5](https://github.com/BlueBrain/axon-synthesis/commit/9ae19c58b900a504d637960254c2081342f57b30))
- Improve graph creation in favored regions (Adrien Berchet - [7408f19](https://github.com/BlueBrain/axon-synthesis/commit/7408f198e47f1aef9b2abd5678ae8af75d0148a0))
- Remove extension parameter and find the corresponding files automatically (Adrien Berchet - [3ee6470](https://github.com/BlueBrain/axon-synthesis/commit/3ee6470180315f40d8d22ae0ca5d902b61e31288))
- Can project multiple tufts in each target population (Rémy Valentin Petkantchin - [9ff53a3](https://github.com/BlueBrain/axon-synthesis/commit/9ff53a3953025d1482f60f2eb3a7d87db9246aa1))
- Add WhiteMatterRecipe.exists() and some formatting (Adrien Berchet - [6e7e6fc](https://github.com/BlueBrain/axon-synthesis/commit/6e7e6fc73990c638d35f94ac43a868abb131b45c))
- Get layer profiles and compute region volumes (Adrien Berchet - [fec9b1b](https://github.com/BlueBrain/axon-synthesis/commit/fec9b1bfcab119e992138e813a36746591ea37ff))
- Add config file (Adrien Berchet - [68ae069](https://github.com/BlueBrain/axon-synthesis/commit/68ae069056670d3cce0159edb8c1c35f0cfdf699))
- Added hemispheres distinction (Rémy Valentin Petkantchin - [350db79](https://github.com/BlueBrain/axon-synthesis/commit/350db79386da23e0140641407e529ce86e83ae99))
- Make the WMR optional for input creation (Adrien Berchet - [118bf50](https://github.com/BlueBrain/axon-synthesis/commit/118bf503ace7a9a36e22ab629269b256ba7af894))
- (mimic preferred region) Allow to keep dummy atlas for later debug (Adrien Berchet - [4156969](https://github.com/BlueBrain/axon-synthesis/commit/4156969d8830637b1ad068842c0946132583cc0b))
- Allow to export edges of the initial and final morphologies (Adrien Berchet - [474b069](https://github.com/BlueBrain/axon-synthesis/commit/474b069e5e5927b1f5939703034b238764ecfe0c))
- Rotate the tufts using the atlas orientations (Adrien Berchet - [b13f28a](https://github.com/BlueBrain/axon-synthesis/commit/b13f28a4d0e9b687f74332c0b51a3a9a80af755f))
- Export more data from the clustering step in debug mode (Adrien Berchet - [840df73](https://github.com/BlueBrain/axon-synthesis/commit/840df73c39d160ec077cf730585abaf42919ea83))
- Add many options in CLI (Adrien Berchet - [9dae736](https://github.com/BlueBrain/axon-synthesis/commit/9dae7361f523534b85bbd4771ee8ff01f480f999))
- Add export for tuft morphologies (Adrien Berchet - [977902f](https://github.com/BlueBrain/axon-synthesis/commit/977902f8ca343817f13cc4ca197a654f950f53b5))
- Improve buffer computation (Adrien Berchet - [598b848](https://github.com/BlueBrain/axon-synthesis/commit/598b8482fe45c54156ec7a653de1fee4ec09cf44))
- Add --atlas-enable/--atlas-disable to be able to run synthesis on mimic inputs (Adrien Berchet - [8c8f813](https://github.com/BlueBrain/axon-synthesis/commit/8c8f8136b59cadf21c7ad203fc980871d93a8b39))
- Add neurite type filter to export_morph_edges (Adrien Berchet - [5eb311b](https://github.com/BlueBrain/axon-synthesis/commit/5eb311b6f5c5aa01451215b053ce273ff1728ce6))
- Export morphologies with NEURON order (Adrien Berchet - [c65b0f8](https://github.com/BlueBrain/axon-synthesis/commit/c65b0f8c7ace7d9a11c2caa4a35f260227b59c1a))
- Remove unifurcations at the end of the synthesis process (Adrien Berchet - [256b956](https://github.com/BlueBrain/axon-synthesis/commit/256b956450238102c1a5455ad9b85ee525ffdde1))

### Fixes

- Fix for Pandas&lt;2.2; fix random numbers; recenter morphs in mimic workflow (Adrien Berchet - [9c60298](https://github.com/BlueBrain/axon-synthesis/commit/9c602981a28e425b6460c2a14ebaca2c215b6d61))
- Handle each clustering config properly (Adrien Berchet - [108635f](https://github.com/BlueBrain/axon-synthesis/commit/108635f61753fa4a64e6bac2f91ee2553a40a279))
- Use proper scales in plots and fix morph recentering issues (Adrien Berchet - [f6ee18b](https://github.com/BlueBrain/axon-synthesis/commit/f6ee18b46e081ad34a4b7a730eb64df46fa1da53))
- The mimic workflow did not produce expected results (Adrien Berchet - [478df02](https://github.com/BlueBrain/axon-synthesis/commit/478df024a0ac7b11d9986d598b5e1d0bccb5f38e))
- Fix edge cases with multiple axons and axons with less that 5 points (Adrien Berchet - [7206e3a](https://github.com/BlueBrain/axon-synthesis/commit/7206e3a1779ea85f01914dd67a6b666ae25a915d))
- Clustering should work for multiple axons and with any neurite order (Adrien Berchet - [e6c0dbe](https://github.com/BlueBrain/axon-synthesis/commit/e6c0dbee61420906994b1565d3f23b8fd53c26da))
- Fix clustering with multiple axons (Adrien Berchet - [ae72a79](https://github.com/BlueBrain/axon-synthesis/commit/ae72a79822f84ca37e0c2dceda29095a32af0edd))
- Graft the tufts to proper sections and propagate rng properly (Adrien Berchet - [25653d7](https://github.com/BlueBrain/axon-synthesis/commit/25653d760e230c6e18d4af0997f5fb82f7e4ec93))
- Fix new edge cases and improve plots to make colors consistent (Adrien Berchet - [4291d8c](https://github.com/BlueBrain/axon-synthesis/commit/4291d8cb682aaa887dc36b07b07b7413f171b421))
- Fix orientations, incorrect source point coordinates and MPI computation (Adrien Berchet - [945bb99](https://github.com/BlueBrain/axon-synthesis/commit/945bb9926f1b8b44c852753b91295eafe6d816ee))
- Fix target positions in atlas and reduce memory for their computation (Adrien Berchet - [f1f0d3f](https://github.com/BlueBrain/axon-synthesis/commit/f1f0d3fe639baf5b48d8dda968e3bcfbfd4e97e4))
- Fix figures for empty morphologies and target points for missing targets (Adrien Berchet - [9949c4e](https://github.com/BlueBrain/axon-synthesis/commit/9949c4ee8a7d9902cb23a84716f6e3dc79a32795))
- Fix input relative path and edge cases with empty morphologies (Adrien Berchet - [f9c5d94](https://github.com/BlueBrain/axon-synthesis/commit/f9c5d94b50c7c3c500f9cc350b14f3282b1e1793))
- Better support of morphologies with no target (Adrien Berchet - [2638c9b](https://github.com/BlueBrain/axon-synthesis/commit/2638c9bcf2e338aefc4df945acc91d76f1505194))
- Fix bbox in graph creation (Adrien Berchet - [5dcab1a](https://github.com/BlueBrain/axon-synthesis/commit/5dcab1af1d78fddd27822f510c03afe009819b41))
- Use proper extensions for JSON files and format them (Adrien Berchet - [6ae9e41](https://github.com/BlueBrain/axon-synthesis/commit/6ae9e4146eaff07c0cea4544fca9a00509476b07))
- Remove points outside the atlas (Adrien Berchet - [4bb0c4a](https://github.com/BlueBrain/axon-synthesis/commit/4bb0c4aef889716d052aac55683f95593da486d9))
- Handle empty morphs in figures (Adrien Berchet - [7cb116d](https://github.com/BlueBrain/axon-synthesis/commit/7cb116dcd33a2b3845e1251a8d798bfbdc191462))
- Fix synthesis workflow in parallel and fix target point check for multiple axons morphologies (Adrien Berchet - [1a00199](https://github.com/BlueBrain/axon-synthesis/commit/1a00199f219831387fb7bb6992cd4a7ff41626f6))
- Fixed hemisphere target points symmetry and added option for different h_axis direction (Rémy Valentin Petkantchin - [106a4a7](https://github.com/BlueBrain/axon-synthesis/commit/106a4a7cef714c7a6c9a6c45f12dde6097d86094))
- Skip morphologies without any axons in mimic workflow (Adrien Berchet - [b313276](https://github.com/BlueBrain/axon-synthesis/commit/b313276d5690885b5439fcbf0f7fffa6c33c0255))
- Fix the target orientation storage (Adrien Berchet - [31706ea](https://github.com/BlueBrain/axon-synthesis/commit/31706ea06ae6b8bfecd7ba40d8415bba594849b7))
- Fix clustering path, HDF5 issue with mode=w and minor fixes (Adrien Berchet - [f6b10e7](https://github.com/BlueBrain/axon-synthesis/commit/f6b10e7203c84a610fc019aa30797e247302ec4a))
- Drop duplicated targets properly (Adrien Berchet - [0a1266c](https://github.com/BlueBrain/axon-synthesis/commit/0a1266c8d5b9d47d68dd895749348d3126d9d187))
- Add missing tasks in the CreateInputs workflow (Adrien Berchet - [5a9606b](https://github.com/BlueBrain/axon-synthesis/commit/5a9606b18492c710c8b3701a2d9d0af5f239ec45))
- Fix parallel partitions in edge case (Adrien Berchet - [ea1cdc7](https://github.com/BlueBrain/axon-synthesis/commit/ea1cdc716fb716010ab01dd988af0d1816e177ad))
- Properly count the number of synthesized morphologies (Adrien Berchet - [75ea4e6](https://github.com/BlueBrain/axon-synthesis/commit/75ea4e6cf3b9c2b5aaac12fe5b951444fd28d416))
- Compute the tuft barcodes properly (Adrien Berchet - [0b5d0c1](https://github.com/BlueBrain/axon-synthesis/commit/0b5d0c1637bbbcf34a1f4d090a62957e87e0c8a2))
- Fix log entry for Steiner Tree solution (Adrien Berchet - [9ddcb6f](https://github.com/BlueBrain/axon-synthesis/commit/9ddcb6fe391288ce5258a2f361b6d5ee902f63df))
- Properly call fillna with inplace=True (Adrien Berchet - [f6b4489](https://github.com/BlueBrain/axon-synthesis/commit/f6b44895ed2115f99b3ff408f7915400507c924d))
- Use axon ID when removing invalid target points (Adrien Berchet - [cdae891](https://github.com/BlueBrain/axon-synthesis/commit/cdae891500850df773ac2e489269c9087988bffb))
- Handle empty list for preferred regions (Adrien Berchet - [5cb2e7c](https://github.com/BlueBrain/axon-synthesis/commit/5cb2e7cfd87176306eed1b22320fd6c954219fa8))
- Reset index after dropping morphs with unknown source brain region (Adrien Berchet - [83c0ca2](https://github.com/BlueBrain/axon-synthesis/commit/83c0ca231dde318d73a47c076031634a250d9a1e))

### Chores And Housekeeping

- Clean research dir (Adrien Berchet - [9fcec1c](https://github.com/BlueBrain/axon-synthesis/commit/9fcec1c8b46ceed2e95600524122da4b93e4e5cf))
- Clean random walk (Adrien Berchet - [8e10fb0](https://github.com/BlueBrain/axon-synthesis/commit/8e10fb0a9cec9b7d5629b51b7ec283a5e36b0bd9))
- Rename cluster-related variables with tuft suffix (Adrien Berchet - [08ded94](https://github.com/BlueBrain/axon-synthesis/commit/08ded9423bb0d1cbc20db4f84e41fd3e5c8515e7))
- Simplify and improve random walk (Adrien Berchet - [26ebf33](https://github.com/BlueBrain/axon-synthesis/commit/26ebf33b1d71a80c0ceece994b7e4fc3fa30a974))
- Cleaning and always store section ID in steiner morph (Adrien Berchet - [88f2744](https://github.com/BlueBrain/axon-synthesis/commit/88f2744995f3fa195a3d0230bff2fc4a35917940))
- Rename favored regions into preferred regions (Adrien Berchet - [523c033](https://github.com/BlueBrain/axon-synthesis/commit/523c033a6609d9fc892676416b7c7602e291787d))
- Replace check_min_max by native validators from attrs (Adrien Berchet - [3f0fc51](https://github.com/BlueBrain/axon-synthesis/commit/3f0fc51d4cf6b81c00de89a0857dba600f41e3b6))
- Improve preferred regions (Adrien Berchet - [511d165](https://github.com/BlueBrain/axon-synthesis/commit/511d1652e5574aef97fe752953fdbf13cee57c47))
- Add more targets to clustering (Adrien Berchet - [ce1b356](https://github.com/BlueBrain/axon-synthesis/commit/ce1b356753cb02fc3737e74574b502195255734e))
- Define and use the CoordsCols class (Adrien Berchet - [1aa807b](https://github.com/BlueBrain/axon-synthesis/commit/1aa807b9c689e7723cf4d03442f265d6de5ee253))
- Minor improvements (Adrien Berchet - [25d3a36](https://github.com/BlueBrain/axon-synthesis/commit/25d3a367bab5b240a32fa61a60ffe6c0152210ad))
- Minor improvements (Adrien Berchet - [d959d96](https://github.com/BlueBrain/axon-synthesis/commit/d959d965d4f1476a427228f3e18f3652757b1a49))
- Move parameter and distribution keys from apical_dendrite to axon (Adrien Berchet - [278428f](https://github.com/BlueBrain/axon-synthesis/commit/278428f16e9b858156736778804f3c09ca3c7ef8))
- Minor cleaning (Adrien Berchet - [896da34](https://github.com/BlueBrain/axon-synthesis/commit/896da3413ecc0d9f3681a1c72ad8258deed19e56))
- Reduce memory for trunk properties and improved figures (Adrien Berchet - [5e84e08](https://github.com/BlueBrain/axon-synthesis/commit/5e84e08bebb219b758529fd5e9e94ab41a868ca0))
- Clean code (Adrien Berchet - [2801ad5](https://github.com/BlueBrain/axon-synthesis/commit/2801ad51b34bd1cb41de89ef60e8a49a33ec77ae))
- Update log entries (Adrien Berchet - [f823cb3](https://github.com/BlueBrain/axon-synthesis/commit/f823cb3cf4af448e4194c541d2be23a386c6520f))
- Improve optional files handling in Clustering (Adrien Berchet - [961bb46](https://github.com/BlueBrain/axon-synthesis/commit/961bb468c8fbda4e6ceec1a4df9413bfc4b98736))
- Bump dir-content-diff (Adrien Berchet - [3324e81](https://github.com/BlueBrain/axon-synthesis/commit/3324e819e84b91fc3876086eabc311d52217f443))
- Fix parameters and distributions (Adrien Berchet - [f015dc8](https://github.com/BlueBrain/axon-synthesis/commit/f015dc8b0b3046e38a8b159b8ab14c9a6b62b6b2))
- Add log entries for source and target points (Adrien Berchet - [0b0d56b](https://github.com/BlueBrain/axon-synthesis/commit/0b0d56b7bbf1b04719dae985359de06191a7bce2))
- Improve brain region mask creation perf (Adrien Berchet - [0efc681](https://github.com/BlueBrain/axon-synthesis/commit/0efc68102ba47c4eeb78c2e39562f096b2098e94))
- Fix package name (Adrien Berchet - [add00dc](https://github.com/BlueBrain/axon-synthesis/commit/add00dcb475e91b33586e6a6a97f7a2771e546ec))
- Remove some Pandas warnings (Adrien Berchet - [124a064](https://github.com/BlueBrain/axon-synthesis/commit/124a064fa944dad6103e963cf98cb929578e4b17))
- Update example scripts (Adrien Berchet - [87c7eb5](https://github.com/BlueBrain/axon-synthesis/commit/87c7eb5c18a20d577f8d106230d2acbaeaf30f00))
- Directly cast the 'grafting_section_id' column (Adrien Berchet - [b66faa8](https://github.com/BlueBrain/axon-synthesis/commit/b66faa8b2f2308d013e39d1a0e9af5d6591baabf))
- Fix language detection (Adrien Berchet - [89f7445](https://github.com/BlueBrain/axon-synthesis/commit/89f7445926dea2264699da0377561bd8041f5996))

### Documentation Changes

- Add figure for preferred region point (Adrien Berchet - [54d7050](https://github.com/BlueBrain/axon-synthesis/commit/54d705065c4e748eb7afd67e37b658978d82ebe9))
- Add CLI and configuration file details (Adrien Berchet - [8f678e9](https://github.com/BlueBrain/axon-synthesis/commit/8f678e97f67f7ab680c1b6fe735ad00d6c6166da))
- Add some general concepts on axon synthesis algorithm (Adrien Berchet - [53a138e](https://github.com/BlueBrain/axon-synthesis/commit/53a138ed202a398e80d4691039ba121f90a2bf34))
- Add input file descriptions (Adrien Berchet - [93dfbe4](https://github.com/BlueBrain/axon-synthesis/commit/93dfbe40379612b43c564a50423831ab88341619))
- Add workflow diagrams and minor improvements (Adrien Berchet - [8e7139a](https://github.com/BlueBrain/axon-synthesis/commit/8e7139a9abe41192cfb19bd546ff67be3570a8bf))
- Add preferred regions in general principles (Adrien Berchet - [23370eb](https://github.com/BlueBrain/axon-synthesis/commit/23370eb0a68d68533dc5db9795c95ab2f7bb3462))

### Refactoring and Updates

- Packaging (Adrien Berchet - [6c631a6](https://github.com/BlueBrain/axon-synthesis/commit/6c631a6bd674bd9cefd6d549808071116c11a76c))
- Fetch WMR works (Adrien Berchet - [91f1d21](https://github.com/BlueBrain/axon-synthesis/commit/91f1d21575a16da808ba8af4c2271fb3d379ea1f))
- Add mimic workflow and improve global config from file in CLI (Adrien Berchet - [1407c46](https://github.com/BlueBrain/axon-synthesis/commit/1407c4636c0808aa9244f43d17f927c334fb84cd))
- Split large modules into smaller ones (Adrien Berchet - [a48db16](https://github.com/BlueBrain/axon-synthesis/commit/a48db1629f9c29d7d7be4c40bb4231b9b2ae8f3f))
- Reorganize CLI and input creation files (Adrien Berchet - [c200363](https://github.com/BlueBrain/axon-synthesis/commit/c20036325a049784a4da3a495230869211f162b5))
- Remove useless files and improve formatting (Adrien Berchet - [1b17f2e](https://github.com/BlueBrain/axon-synthesis/commit/1b17f2e45c3f27086c9252ccf942ccf300a58de3))
- Move functions from clutering to sub-modules (Adrien Berchet - [f9ebab5](https://github.com/BlueBrain/axon-synthesis/commit/f9ebab5261ae5097585359440b4c5ee964dbd8f5))
- Split CreateTuftTerminalProperties run method into smaller functions (Adrien Berchet - [5ce14f4](https://github.com/BlueBrain/axon-synthesis/commit/5ce14f4ae4116f25d869b2e9b31b27f62f52a1a3))
- Reduce run() method size in the ClusterTerminals and FindTargetPoints tasks (Adrien Berchet - [ef288e9](https://github.com/BlueBrain/axon-synthesis/commit/ef288e90f77acc4a9b664f1b4c8d545a4d6781d7))
- Move functions from create_graph to other sub-modules (Adrien Berchet - [ea4958a](https://github.com/BlueBrain/axon-synthesis/commit/ea4958a1f6c322c127fb4a1877d60f133d18d19e))
- Simplify the code and fix the tests (Adrien Berchet - [3b81b3b](https://github.com/BlueBrain/axon-synthesis/commit/3b81b3bd0c18d7214771395e324d278c2790d676))
- Extract the trunk properties into a new task (Adrien Berchet - [98e8a2b](https://github.com/BlueBrain/axon-synthesis/commit/98e8a2bd0aed0724d07258377797b1215fe2dd59))
- Clustering works, at least the sphere-parent method (Adrien Berchet - [aeee93d](https://github.com/BlueBrain/axon-synthesis/commit/aeee93da9e5185ebfe503432175c5e3c9aaefa51))
- Move WMR processing (Adrien Berchet - [167119d](https://github.com/BlueBrain/axon-synthesis/commit/167119de928258541dde60bbdaf4c1480a29e093))
- Move plot of AddTufs task into an external function (Adrien Berchet - [7022945](https://github.com/BlueBrain/axon-synthesis/commit/7022945c38a8bb8e6767ae492573587f7818821e))

### Changes to Test Assests

- Use not resampled morphologies (Adrien Berchet - [807ccdf](https://github.com/BlueBrain/axon-synthesis/commit/807ccdf5468650dbb26832cd5b39c3f243124302))
- Fix the mimic workflow test and remove a few warnings (Adrien Berchet - [804a48c](https://github.com/BlueBrain/axon-synthesis/commit/804a48c8f451673693fac527b2793851dc66a2be))
- Add missing files and minor fixes (Adrien Berchet - [9487823](https://github.com/BlueBrain/axon-synthesis/commit/94878238cfca4754369f0e288f121fc0c510858c))
- Add a simple test case and fix revealed bugs (Adrien Berchet - [5384d4a](https://github.com/BlueBrain/axon-synthesis/commit/5384d4a5122ae69a6f45c6b253c27dd0f138f418))
- Fix tests and related issues (Adrien Berchet - [13ce8dd](https://github.com/BlueBrain/axon-synthesis/commit/13ce8dd3faa8ae72b105b192bf4acb644e6515c5))
- Add a test for mimic case (Adrien Berchet - [a033614](https://github.com/BlueBrain/axon-synthesis/commit/a03361479d8300c0a5a791f17adb4e151e0927e1))
- Fix tolerance for Morphology diffs (Adrien Berchet - [a7b0a04](https://github.com/BlueBrain/axon-synthesis/commit/a7b0a04229e6515cc514794d75c3e8f16956f3e7))
- Update SteinerTreeSolutions data (Adrien Berchet - [1b3e01c](https://github.com/BlueBrain/axon-synthesis/commit/1b3e01c09223691e54ec44d42ca408336dfce656))

### Tidying of Code eg Whitespace

- Format according to Ruff (Adrien Berchet - [ae36376](https://github.com/BlueBrain/axon-synthesis/commit/ae363762c03ef91c67204bc0c3efb3f1609c68ad))
- Setup mypy checks and add type hints (Adrien Berchet - [d6b435f](https://github.com/BlueBrain/axon-synthesis/commit/d6b435f80789ce99fc3cc0b9ece0d273491541fe))
- Fix many linting error (Adrien Berchet - [e7ffad1](https://github.com/BlueBrain/axon-synthesis/commit/e7ffad168b13c53436deb37a3b606a06feb571e4))
- Setup mypy checks and add type hints (Adrien Berchet - [b9b0bac](https://github.com/BlueBrain/axon-synthesis/commit/b9b0bac2d607292fc59a8e216c836bfa16d5652c))
- Lint (Adrien Berchet - [60ea4f9](https://github.com/BlueBrain/axon-synthesis/commit/60ea4f91878d657d9561a8baa27c2ec2d5e3a0ee))

### Performance Improvements

- Target points are selected muck quicker (Adrien Berchet - [fc9be79](https://github.com/BlueBrain/axon-synthesis/commit/fc9be796ce0493e8c056c7cdc4bf39e32e5a349a))

### CI Improvements

- Fix gitlab config, lint, mypy, ... (Adrien Berchet - [38d8e33](https://github.com/BlueBrain/axon-synthesis/commit/38d8e33b385819c3cdbd2da387ba38372dc05ccc))
- Bump versions in pre-commit config (Adrien Berchet - [1447a46](https://github.com/BlueBrain/axon-synthesis/commit/1447a46e55acbebf622fa000c631e6999ff9c123))
- Fix gitlab configuration (Adrien Berchet - [c96fecb](https://github.com/BlueBrain/axon-synthesis/commit/c96fecbd554b9ca6772546303fd3ce5e5d87f48e))
- Use build instead setup.py (Adrien Berchet - [914bf25](https://github.com/BlueBrain/axon-synthesis/commit/914bf25be654f12e549ef24f0d66cbba7f044584))

### General Changes

- Fix source and target points (Adrien Berchet - [7f75787](https://github.com/BlueBrain/axon-synthesis/commit/7f75787d170fb98196f64c2a02444310575188a2))
- Add clustering by regions (Adrien Berchet - [c243224](https://github.com/BlueBrain/axon-synthesis/commit/c24322454b7c2d42f51747bc0b90695c89b161c2))
- Black and isort (Adrien Berchet - [253804e](https://github.com/BlueBrain/axon-synthesis/commit/253804e64e19b8734a7df0c0db7107ee08dab224))
- Improve logger and add trunk grafting (Adrien Berchet - [eb3e3ac](https://github.com/BlueBrain/axon-synthesis/commit/eb3e3ac71f676f56c5b659814f366d3bd3c5bc4f))
- Add graph creation and Steiner Tree solution (Adrien Berchet - [f43e414](https://github.com/BlueBrain/axon-synthesis/commit/f43e4144acde535c2a698e5efe1cfa9e2bf41ec9))
- Add new tasks to synthesize axons from white matter recipe (Adrien Berchet - [a809e51](https://github.com/BlueBrain/axon-synthesis/commit/a809e517c93b1b5293e5516f69a03cdc183d92d1))
- Compute path length of the tufts from the white matter recipe (Adrien Berchet - [f59f16d](https://github.com/BlueBrain/axon-synthesis/commit/f59f16daf0cc3a20aee37b01e9d12dd04b3cb19a))
- Refactor several classes to make the code simpler (Adrien Berchet - [721543a](https://github.com/BlueBrain/axon-synthesis/commit/721543aa6bf728d61c46fda7d2f18e8fc849b6c8))
- Improve clustrering and use it in later tasks instead of raw morphologies (Adrien Berchet - [42b00a5](https://github.com/BlueBrain/axon-synthesis/commit/42b00a57076b49a69cef6f3e540eafae0d45a6e7))
- Update workflows to synthesize axons in vacuum or using the white matter recipe (Adrien Berchet - [760bfbc](https://github.com/BlueBrain/axon-synthesis/commit/760bfbc509615fc5eda668fbd6f8d1dc30ddb8c2))
- Rework clustering params format and create a class to store results (Adrien Berchet - [b14d8be](https://github.com/BlueBrain/axon-synthesis/commit/b14d8be06dec8a30d2b27759ac716ab2429d2ffc))
- Improve clustering and tuft statistics, add options in graph creation and fix tuft generation (Adrien Berchet - [6331b2d](https://github.com/BlueBrain/axon-synthesis/commit/6331b2d723743b0841db4bc8bbaccf42e257f498))
- Add new clustering method and improve graph creation (Adrien Berchet - [23de2cc](https://github.com/BlueBrain/axon-synthesis/commit/23de2cc9d90ba19ac54e012ae64f8059d368d5cb))
- Add a tuft to each terminal (Adrien Berchet - [1c588e3](https://github.com/BlueBrain/axon-synthesis/commit/1c588e359e649b25def470007e314daeaccfa557))
- Add score matrix as validation tool (Adrien Berchet - [82e61a3](https://github.com/BlueBrain/axon-synthesis/commit/82e61a3ce394560fff28bc0f74af22c7407863ad))
- Create morphologies from Steiner solutions and plot the morphologies with plotly_helper (Adrien Berchet - [1b04130](https://github.com/BlueBrain/axon-synthesis/commit/1b041300717a39ee79715eb9825ae1b3c0647fb6))
- Improve clustering and statistics (Adrien Berchet - [b202539](https://github.com/BlueBrain/axon-synthesis/commit/b202539c587681b88f703ee840e4b37aa000d94d))
- Add clustering and use luigi workflows (Adrien Berchet - [0c0d662](https://github.com/BlueBrain/axon-synthesis/commit/0c0d6627ebfc359ce54acb61aab17ed52f37834f))
- Simplify some classes (Adrien Berchet - [eafeaca](https://github.com/BlueBrain/axon-synthesis/commit/eafeacade2b58a410d08301144bf66cebbab5957))
- Add Steiner computation and solution export (Adrien Berchet - [c2a2d57](https://github.com/BlueBrain/axon-synthesis/commit/c2a2d57075ed872e9234df883602805f6ada88f1))
- Add some helpers for atlas and white matter repice (Adrien Berchet - [8197b60](https://github.com/BlueBrain/axon-synthesis/commit/8197b600f3d4ca81b477c5c4b6fc91b61323fc90))
- Add graph creation step for Steiner Tree computation (Adrien Berchet - [5d8c6a6](https://github.com/BlueBrain/axon-synthesis/commit/5d8c6a6b80da7ad7e16cab1996d36e57dcb82ae3))
- Cleaning and start statistics (Adrien Berchet - [eae593a](https://github.com/BlueBrain/axon-synthesis/commit/eae593a7f4e8b3c507b5fa39d19cf795fbc1da14))
- Minor cleaning (Adrien Berchet - [4a02f6f](https://github.com/BlueBrain/axon-synthesis/commit/4a02f6f104d8007f1c68ead0a67d18d58f70c317))
- Minor refactoring and add features to atlas (Adrien Berchet - [ffabf3c](https://github.com/BlueBrain/axon-synthesis/commit/ffabf3c4baaf0c5ba5c09a7e6351b8d9304c32c6))
- Start building the tufts (Adrien Berchet - [44f82ff](https://github.com/BlueBrain/axon-synthesis/commit/44f82ffc7013127b20cb359a3b8b4bfab5da48c8))
- Add statistics (Adrien Berchet - [00a524b](https://github.com/BlueBrain/axon-synthesis/commit/00a524b23d14be84d7bbcccd36a0ac6a19e41211))
- Initial commit (Adrien Berchet - [3afee78](https://github.com/BlueBrain/axon-synthesis/commit/3afee7852bed81db5ebc793519ae518fd5419d02))
- Add missing file (Adrien Berchet - [963f5c1](https://github.com/BlueBrain/axon-synthesis/commit/963f5c122b35d9ce97f3b72b3d86fb60d6ca90c3))
- Repair and plot raw morphologies (Adrien Berchet - [db16cbb](https://github.com/BlueBrain/axon-synthesis/commit/db16cbbac926e512550ee61f31fd3a1290d61b57))
- Add random choice of tufts from the input ones (Adrien Berchet - [c2576f2](https://github.com/BlueBrain/axon-synthesis/commit/c2576f2a4a6d5125543ac119e89d0ac6a40484cb))
- Add task to automatically fetch the White Matter Recipe file (Adrien Berchet - [6194ce3](https://github.com/BlueBrain/axon-synthesis/commit/6194ce3ef779a19e659e08c1fd6a7b5820a0eed6))
- Fix clustering and graph creation and use OutputLocalTarget (Adrien Berchet - [59015d3](https://github.com/BlueBrain/axon-synthesis/commit/59015d3923b59a7f0c1c9d4fbd15da3146bd6c1b))
- Add intermediate points before Voronoi points (Adrien Berchet - [8bc1345](https://github.com/BlueBrain/axon-synthesis/commit/8bc13458d782b71c4b7f5f99e7accef86fa69e04))
- Correct the mean length of the tuft by the trunk length (Adrien Berchet - [aa684cf](https://github.com/BlueBrain/axon-synthesis/commit/aa684cf2f6b8ff18be3687e43db8cc03fe67aaf4))
- Fix morph export paths (Adrien Berchet - [aba4a57](https://github.com/BlueBrain/axon-synthesis/commit/aba4a5735f71a8a37834a6c65a213e2fc01c7bb1))
- Can add random points before the Voronoi process (Adrien Berchet - [b050982](https://github.com/BlueBrain/axon-synthesis/commit/b050982fe168e87b7cc7fdfa8cd5fa0a0f69c678))
- Fix distrs and params for NeuroTS&gt;=3.2 (Adrien Berchet - [da83578](https://github.com/BlueBrain/axon-synthesis/commit/da83578bc36c019608a1675505d230522d5f8870))
- Fixes and add new configs (Adrien Berchet - [6ae53e9](https://github.com/BlueBrain/axon-synthesis/commit/6ae53e98ffae02adc8cb9f19fb6881f90793541c))
- Add new task to create specific inputs (Adrien Berchet - [57b61f4](https://github.com/BlueBrain/axon-synthesis/commit/57b61f445abca50e21f8bb2849cdc4beb422de38))
- Add requirements.txt (Adrien Berchet - [38142e0](https://github.com/BlueBrain/axon-synthesis/commit/38142e0afea016c0cf549e2570ec60f3495a1216))
- Simplify some classes (Adrien Berchet - [6955e4a](https://github.com/BlueBrain/axon-synthesis/commit/6955e4a3c95bd9c91eb8bd46d27ac0a1341db5f8))

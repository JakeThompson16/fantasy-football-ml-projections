# fantasy-football-ml-projections
Uses Gradient Boosting Decision Tree (GBDT) models to project fantasy football player scores. Currently supports running backs and wide receivers, with additional positions planned. Models are trained on a variety of common, advanced, and invented stats aggregated from differing amounts of games.

Invented Statistics Include:
Running Backs:
- Weighted Rushes (both for player and against defense)
    - If rush is not in redzone it counts as 1
    - If rush is in redzone it counts as 20 / yardline of rush
    - Captures a players total rushing opportunity inflated by proximity to       endzone
- Weighted Targets:
    - Same as weighted rushes, but for passing plays
- Red Zone Carry Capitalization Score
    - Total redzone carry touchdowns / carry weight per redzone carry
    - Explains a players ability to capitalize on high value carries
    - Highly valued by model
- Red Zone Carry Capitalization Score
    - Same as Red Zone Carry Capitalization Score, but for passing plays

Wide Receivers:
- Big Play Conversion Rate
    - Rate at which a player catches passes of 20+ air yards
- Receiver Quality Score
    - Takes a players average depth of target divided by the max average           depth of target by any WR in that season
    - Takes a players average receptions per game divided by the max average       receptions per game by any WR in that season
    - Takes the sum of both of those metrics
    - Aims to explain a receivers usage and quality of opportunity compared        to their peers
    - Inspired by + stats (OPS+, ERA+, etc.) in baseball that base metrics         on a comparison to all other players in that season
- Boom Score
    - receiving td's per game * 4 + redzone targets per game + big play            conversion rate * 3
    - Aims to capture a players opportunity and capitalization of big plays        that heavily effect fantasy outputs
    - Highly valued by model
- Target Quality Score
    - Average depth of target + (average separation)^2
    - Aims to explain a players opportunity to gain yards by air yards in         conjunction with their opportunity to create yards after catch
- Weighted Target Score
    - Target share + air yards share * 1.5
    - Aims to explain a receivers volume and the quality of the volume
- Target Value Added
    - receiving epa / targets
    - Essentially EPA per target
    - Highly valued by model
- Yard Opportunity Capitalization Score
    - Catch percentage * average depth of target + receiving YAC
    - Aims to explain a players ability to create yards through all facets         of play

Metrics:
Running Back Model:
Recently identified source of leakage artifically inflating r^2 and MAE

Wide Receiver Model:
R²: 0.2671507862990038
MAE: 4.9060610798892474

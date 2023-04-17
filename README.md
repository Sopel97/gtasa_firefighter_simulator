# GTA San Andreas Firefighter Simulator

This is a faithful reimplementation of the GTA San Andreas in-game firefighter submission generator. It also exposes the in-game node data for pathfinding on roads (though, as of 2023-04-17, it requires more work to properly reflect common paths that don't follow the road lanes)

## TODO

### Model and algorithms

- IMPORTANT: Improve the connectivity in the graph for pathfinding. Currently it only follows road lanes, which is not optimal in most cases.
- Figure out a good metric for including death-warps
- Figure out if it's possible to take the driving model into account when pathfinding (for example the time needed for a U-turn)
- Figure out better metrics. Average doesn't give the whole picture. For example, minimum could be better for IL hunting, some n% percentile for consistency)
- Research better heuristics for guiding simulation of all the levels at once?

### Presentation

- Smoothen the heatmap
- Add interactive mode?

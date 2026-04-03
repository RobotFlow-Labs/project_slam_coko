# PRD-06: ROS2 Integration

> Module: SLAM-COKO | Priority: P1  
> Depends on: PRD-01, PRD-02, PRD-03, PRD-05  
> Status: ✅ Complete

## Objective

Wrap the Coko-SLAM runtime in ROS2 nodes and launch files so ANIMA can replay or stream RGB-D data through the paper pipeline.

## Context (from paper)

The paper is evaluated offline, but the target ANIMA stack needs ROS2 message I/O while preserving the same submap and loop-closure logic.

**Paper reference:** same runtime graph as Fig. 2, adapted to ROS2 transport

## Acceptance Criteria

- [ ] ROS2 node ingests synchronized RGB and depth topics
- [ ] ROS2 node publishes submap summaries, optimized poses, and merged map status
- [ ] Launch file supports replay and live RGB-D modes
- [ ] A thin adapter preserves the paper’s keyframe/submap logic
- [ ] Test: `uv run pytest tests/ros2/test_message_adapters.py -v` passes

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|---|---|---|---|
| `src/anima_slam_coko/ros2/bridge_node.py` | ROS2 runtime bridge | Fig. 2 | ~220 |
| `src/anima_slam_coko/ros2/message_adapters.py` | ROS image/depth conversion | — | ~140 |
| `src/anima_slam_coko/ros2/launch/slam_coko.launch.py` | launch entrypoint | — | ~90 |
| `tests/ros2/test_message_adapters.py` | adapter tests | — | ~100 |

## Architecture Detail (from paper)

### Inputs
- `/camera/color/image_raw`
- `/camera/depth/image_rect`
- `/camera/info`

### Outputs
- `/slam_coko/submap`
- `/slam_coko/optimized_pose`
- `/slam_coko/status`

### Algorithm
```python
class SlamCokoBridge(Node):
    def on_rgbd(self, rgb_msg, depth_msg, info_msg):
        frame = self.adapter.to_frame(rgb_msg, depth_msg, info_msg)
        self.runtime.step(frame)
```

## Dependencies

```toml
rclpy = "*"
sensor-msgs-py = "*"
```

## Data Requirements

| Asset | Size | Path | Download |
|---|---|---|---|
| ROS2 bag fixture | small | `tests/fixtures/ros2/` | generated locally |

## Test Plan

```bash
uv run pytest tests/ros2/test_message_adapters.py -v
```

## References

- Depends on: PRD-02, PRD-03, PRD-05
- Feeds into: PRD-07

"""Step 1 checkpoint: PyBullet world spins up with a working tick loop.

What you should see:
    - A GUI window with a ground plane and default lighting
    - Console prints from a 1 Hz callback, once per simulated second
    - The window closes cleanly after 5 seconds

If any of that fails, world.py has a bug we need to fix before moving on.
"""

from sim.world import World, WorldConfig


def main() -> None:
    world = World(WorldConfig(gui=True))
    world.reset()

    # Register a 1 Hz callback so we can see the scheduler working.
    def heartbeat(sim_time_s: float) -> None:
        print(f"[t={sim_time_s:5.2f}s] heartbeat (step {world.step_count})")

    world.register_callback(rate_hz=1.0, fn=heartbeat)

    print("Running for 5 simulated seconds...")
    world.run(duration_s=5.0)
    print(f"Done. Final sim_time={world.sim_time:.2f}s, "
          f"steps={world.step_count}")
    world.close()


if __name__ == "__main__":
    main()

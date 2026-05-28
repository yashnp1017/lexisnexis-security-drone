"""
LexisNexis Autonomous Security Drone System
============================================
Entry point. Orchestrates sensor fusion, face detection,
threat scoring, REST API alert dispatch, and mission control.

Usage:
    python main.py                 # demo mode (no hardware)
    python main.py --mode full     # full system with hardware
    python main.py --mode api      # REST API server only
"""

import argparse
import asyncio
import logging

from src.mission.controller import MissionController
from src.api.server import start_api_server
from config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run_full_system(settings):
    controller = MissionController(settings)
    api_task = asyncio.create_task(start_api_server(controller, settings))
    try:
        await controller.run()
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
    finally:
        api_task.cancel()
        await controller.shutdown()


async def run_demo(settings):
    from src.mission.demo import run_demo_mission
    await run_demo_mission(settings)


def main():
    parser = argparse.ArgumentParser(description="LexisNexis Autonomous Security Drone")
    parser.add_argument("--mode", choices=["full", "demo", "api"], default="demo")
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    settings = Settings.from_yaml(args.config)
    if args.mode == "demo":
        asyncio.run(run_demo(settings))
    elif args.mode == "api":
        asyncio.run(start_api_server(None, settings))
    else:
        asyncio.run(run_full_system(settings))


if __name__ == "__main__":
    main()

from modal import method, gpu, enter
import json
import traceback
import numpy as np
import os
from typing import Dict, List, Any, Optional

from .deploy_config import get_scaledown_window, get_gpu_config, get_memory_mb
from .modal_config import app, unav_image, volume, gemini_secret, middleware_secret
from .destinations_service import get_destinations_list_impl
from .logic import (
    run_planner,
    run_localize_user,
    run_init_middleware,
    run_init_cpu_components,
    run_init_gpu_components,
    run_monkey_patch_localizer_methods,
    run_monkey_patch_pose_refinement,
    run_monkey_patch_feature_extractors,
    run_monkey_patch_matching_and_ransac,
    run_construct_mock_localization_output,
    run_set_navigation_context,
    run_safe_serialize,
)


@app.cls(
    image=unav_image,
    volumes={"/root/UNav-IO": volume},
    gpu=get_gpu_config(),
    enable_memory_snapshot=False,
    memory=get_memory_mb(),
    timeout=600,
    scaledown_window=get_scaledown_window(),
    secrets=[gemini_secret, middleware_secret],
)
class UnavServer:
    # Initialize session storage for user contexts
    user_sessions: Dict[str, Any] = {}
    tracer: Optional[Any] = None
    _middleware_init_pending: bool = False

    def _gpu_available(self) -> bool:
        return True

    def _configure_middleware_tracing(self):
        pass

    @enter(snap=False)
    def initialize_middleware(self):
        """Delegated to logic.init"""
        run_init_middleware(self)

    @enter(snap=False)
    def initialize_cpu_components(self):
        """Delegated to logic.init"""
        run_init_cpu_components(self)

    @enter(snap=False)
    def initialize_gpu_components(self):
        """Delegated to logic.init"""
        run_init_gpu_components(self)

    def _monkey_patch_localizer_methods(self, localizer, method_names=None):
        """Delegated to logic.init"""
        run_monkey_patch_localizer_methods(self, localizer, method_names)

    def _monkey_patch_pose_refinement(self):
        """Delegated to logic.init"""
        run_monkey_patch_pose_refinement(self)

    def _monkey_patch_feature_extractors(self):
        """Delegated to logic.init"""
        run_monkey_patch_feature_extractors(self)

    def _monkey_patch_matching_and_ransac(self):
        """Delegated to logic.init"""
        run_monkey_patch_matching_and_ransac(self)

    def ensure_gpu_components_ready(self):
        """
        Ensure GPU components are initialized before processing requests.
        This is called by methods that need the localizer.
        """
        if not hasattr(self, "gpu_components_initialized"):
            print("🔧 GPU components not ready, initializing now...")
            self.initialize_gpu_components()

    def get_session(self, user_id: str) -> dict:
        """Get or create user session"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        return self.user_sessions[user_id]

    def update_session(self, user_id: str, updates: dict):
        """Update user session with new data"""
        session = self.get_session(user_id)
        session.update(updates)

    def _construct_mock_localization_output(
        self,
        x: float,
        y: float,
        angle: float,
        place: str,
        building: str,
        floor: str,
    ) -> dict:
        return run_construct_mock_localization_output(
            x=x,
            y=y,
            angle=angle,
            place=place,
            building=building,
            floor=floor,
        )

    def set_navigation_context(
        self,
        user_id: str,
        dest_id: str,
        target_place: str,
        target_building: str,
        target_floor: str,
        unit: str = "meter",
        language: str = "en",
    ):
        return run_set_navigation_context(
            server=self,
            user_id=user_id,
            dest_id=dest_id,
            target_place=target_place,
            target_building=target_building,
            target_floor=target_floor,
            unit=unit,
            language=language,
        )

    @method()
    def start_server(self):
        import json
        import logging

        """
        Initializes and starts the serverless instance.
    
        This function helps in reducing the server response time for actual requests by pre-warming the server. 
        By starting the server in advance, it ensures that the server is ready to handle incoming requests immediately, 
        thus avoiding the latency associated with a cold start.
        """
        print("UNAV Container started...")
        logging.info("UNav server initialization requested")

        # Create span for server start if tracer available
        if hasattr(self, "tracer") and self.tracer:
            with self.tracer.start_as_current_span("start_server_span") as span:
                response = {"status": "success", "message": "Server started."}
                logging.info("Server warmup completed successfully")
                return json.dumps(response)
        else:
            response = {"status": "success", "message": "Server started."}
            return json.dumps(response)

    @method()
    def get_destinations_list(
        self,
        floor="6_floor",
        place="New_York_City",
        building="LightHouse",
        enable_multifloor: bool = False,
    ):
        """
        Get destinations for a specific place, building, and floor.
        Loads places on demand for fast startup.
        """
        return get_destinations_list_impl(
            server=self,
            floor=floor,
            place=place,
            building=building,
            enable_multifloor=enable_multifloor,
        )

    @method()
    def planner(
        self,
        session_id: str,
        base_64_image,
        destination_id: str,
        place: str,
        building: str,
        floor: str,
        top_k: int = None,
        unit: str = "meter",
        language: str = "en",
        refinement_queue: dict = None,
        is_vlm_extraction_enabled: bool = False,
        enable_multifloor: bool = True,
        should_use_user_provided_coordinate: bool = False,
        x: float = None,
        y: float = None,
        angle: float = None,
    ):
        """Full localization and navigation pipeline."""
        return run_planner(
            self,
            session_id=session_id,
            base_64_image=base_64_image,
            destination_id=destination_id,
            place=place,
            building=building,
            floor=floor,
            top_k=top_k,
            unit=unit,
            language=language,
            refinement_queue=refinement_queue,
            is_vlm_extraction_enabled=is_vlm_extraction_enabled,
            enable_multifloor=enable_multifloor,
            should_use_user_provided_coordinate=should_use_user_provided_coordinate,
            x=x,
            y=y,
            angle=angle,
        )

    @method()
    def localize_user(
        self,
        session_id: str,
        base_64_image,
        place: str,
        building: str,
        floor: str,
        top_k: int = None,
        refinement_queue: dict = None,
        enable_multifloor: bool = True,
    ):
        """Localize user position without navigation planning."""
        return run_localize_user(
            self,
            session_id=session_id,
            base_64_image=base_64_image,
            place=place,
            building=building,
            floor=floor,
            top_k=top_k,
            refinement_queue=refinement_queue,
            enable_multifloor=enable_multifloor,
        )

    def get_user_session(self, user_id: str):
        """Get current user session data"""
        try:
            session = self.get_session(user_id)
            return {"status": "success", "session": run_safe_serialize(session)}
        except Exception as e:
            return {"status": "error", "message": str(e), "type": type(e).__name__}

    def clear_user_session(self, user_id: str):
        """Clear user session data"""
        try:
            if user_id in self.user_sessions:
                del self.user_sessions[user_id]
            return {
                "status": "success",
                "message": f"Session cleared for user {user_id}",
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "type": type(e).__name__}

from modal import method
from typing import Any, Dict, Optional

from .deploy_config import get_memory_mb, get_gpu_config, get_scaledown_window
from .destinations_service import get_destinations_list_impl
from .modal_config import app, gemini_secret, middleware_secret, unav_image, volume
from .server_methods.init_methods import (
    _configure_middleware_tracing,
    _gpu_available,
    initialize_cpu_components,
    initialize_gpu_components,
    initialize_middleware,
)
from .server_methods.monkey_patch_methods import (
    _monkey_patch_feature_extractors,
    _monkey_patch_lightglue,
    _monkey_patch_localizer_methods,
    _monkey_patch_matching_and_ransac,
    _monkey_patch_pose_refinement,
)
from .server_methods.navigation_methods import localize_user, planner
from .server_methods.output_methods import (
    _safe_serialize,
    clear_user_session,
    convert_navigation_to_trajectory,
    generate_nav_instructions_from_coordinates,
    get_user_session,
    run_vlm_on_image,
    unav_navigation_simple,
)
from .server_methods.places_methods import (
    _get_fallback_places,
    ensure_gpu_components_ready,
    ensure_maps_loaded,
    get_places,
)
from .server_methods.session_methods import (
    _construct_mock_localization_output,
    get_session,
    set_navigation_context,
    update_session,
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

    # Place/map methods
    get_places = get_places
    _get_fallback_places = _get_fallback_places
    ensure_maps_loaded = ensure_maps_loaded
    ensure_gpu_components_ready = ensure_gpu_components_ready

    # Session/context methods
    get_session = get_session
    update_session = update_session
    _construct_mock_localization_output = _construct_mock_localization_output
    set_navigation_context = set_navigation_context

    # Output/serialization methods
    _safe_serialize = _safe_serialize
    generate_nav_instructions_from_coordinates = generate_nav_instructions_from_coordinates
    get_user_session = get_user_session
    clear_user_session = clear_user_session
    unav_navigation_simple = unav_navigation_simple
    convert_navigation_to_trajectory = convert_navigation_to_trajectory
    run_vlm_on_image = run_vlm_on_image

    # Monkey patch methods
    _monkey_patch_localizer_methods = _monkey_patch_localizer_methods
    _monkey_patch_pose_refinement = _monkey_patch_pose_refinement
    _monkey_patch_feature_extractors = _monkey_patch_feature_extractors
    _monkey_patch_lightglue = _monkey_patch_lightglue
    _monkey_patch_matching_and_ransac = _monkey_patch_matching_and_ransac

    # Initialization methods
    _gpu_available = _gpu_available
    _configure_middleware_tracing = _configure_middleware_tracing
    initialize_middleware = initialize_middleware
    initialize_cpu_components = initialize_cpu_components
    initialize_gpu_components = initialize_gpu_components

    # Main navigation methods
    planner = planner
    localize_user = localize_user

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
            with self.tracer.start_as_current_span("start_server_span"):
                response = {"status": "success", "message": "Server started."}
                logging.info("Server warmup completed successfully")
                return json.dumps(response)

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

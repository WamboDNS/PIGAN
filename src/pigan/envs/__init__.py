"""
Adversarial Prompt Injection Training Environment.

A multi-agent environment for training prompt injection attack (M) and defense (D)
using PrimeIntellect Sandboxes for flag-based success detection.

Example usage:
    from adversarial_robustness.envs import (
        PromptInjectionEnv,
        SandboxWrapper,
        SandboxConfig,
        AliceSimulator,
        BobSimulator,
        SimulatorConfig,
        create_simulators,
        get_scenarios_by_difficulty,
        RewardConfig,
    )

    # Setup sandbox
    sandbox = SandboxWrapper(SandboxConfig())
    sandbox.start()

    # Create simulators (using vLLM client)
    alice, bob = create_simulators(
        client=vllm_client,
        chat_template=chat_template,
        sandbox=sandbox,
        model="meta-llama/Llama-3.1-8B-Instruct",
    )

    # Create environment
    env = PromptInjectionEnv(
        alice=alice,
        bob=bob,
        sandbox=sandbox,
        scenarios=get_scenarios_by_difficulty("easy"),
    )

    # Run with MultiAgentProtocol
    protocol = MultiAgentProtocol(agents={"M": m_agent, "D": d_agent})
    rollouts = await protocol.run(env=env, max_steps=100)
"""

# Core environment
from .env import PromptInjectionEnv

# Sandbox integration
from .sandbox import SandboxConfig, SandboxWrapper, MockSandboxWrapper, create_sandbox

# Scenarios and configuration
from .scenario import (
    BobToolConfig,
    CurriculumConfig,
    InjectionScenario,
    DEFAULT_ALICE_SYSTEM_PROMPT,
    DEFAULT_BOB_SYSTEM_PROMPT,
    DEFAULT_M_SYSTEM_PROMPT,
    DEFAULT_D_SYSTEM_PROMPT,
)

# Simulators
from .simulation import (
    AliceSimulator,
    BobSimulator,
    SimulatorConfig,
    create_simulators,
)

# Detection utilities
from .detection import (
    check_flag_leakage,
    extract_xml_tag,
    parse_d_classification,
    parse_m_output,
    parse_m_injection,
)

# Rewards (sparse terminal)
from .rewards import (
    RewardConfig,
    StepRewards,
    EpisodeOutcome,
    EpisodeSummary,
    compute_step_rewards,
    check_information_leak,
)

# Scenario presets
from .scenarios import (
    ALL_SCENARIOS,
    EASY_SCENARIOS,
    HARD_SCENARIOS,
    MEDIUM_SCENARIOS,
    get_scenarios_by_difficulty,
)

__all__ = [
    # Core
    "PromptInjectionEnv",
    # Sandbox
    "SandboxConfig",
    "SandboxWrapper",
    "MockSandboxWrapper",
    "create_sandbox",
    # Scenarios
    "BobToolConfig",
    "CurriculumConfig",
    "InjectionScenario",
    "DEFAULT_ALICE_SYSTEM_PROMPT",
    "DEFAULT_BOB_SYSTEM_PROMPT",
    "DEFAULT_M_SYSTEM_PROMPT",
    "DEFAULT_D_SYSTEM_PROMPT",
    # Simulators
    "AliceSimulator",
    "BobSimulator",
    "SimulatorConfig",
    "create_simulators",
    # Detection
    "check_flag_leakage",
    "extract_xml_tag",
    "parse_d_classification",
    "parse_m_output",
    "parse_m_injection",
    # Rewards (sparse terminal)
    "RewardConfig",
    "StepRewards",
    "EpisodeOutcome",
    "EpisodeSummary",
    "compute_step_rewards",
    "check_information_leak",
    # Presets
    "ALL_SCENARIOS",
    "EASY_SCENARIOS",
    "MEDIUM_SCENARIOS",
    "HARD_SCENARIOS",
    "get_scenarios_by_difficulty",
]

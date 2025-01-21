from dataclasses import dataclass

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal
from omegaconf import OmegaConf
from rich.status import Status
from .utils.config import (
    load_task_desc,
    prep_agent_workspace,
    save_run,
    _load_cfg,
    prep_cfg,
)


@dataclass
class Solution:
    code: str
    valid_metric: float


class Experiment:

    def __init__(
        self,
        data_dir: str,
        goal: str,
        eval: str | None = None,
        default_model: str | None = None,
        code_model: str | None = None,
        feedback_model: str | None = None,
        report_model: str | None = None,
        workspace_dir: str | None = None,
        exp_name: str | None = None,
        log_dir: str | None = None,
        preprocess_data: bool | None = None,
        copy_data: bool | None = None,
        generate_report: bool | None = None,
    ):
        """Initialize a new experiment run.

        Args:
            data_dir: Path to the directory containing the data files.
            goal: Description of the goal of the task.
            eval: Optional description of the preferred way for the agent to
                evaluate its solutions.
            default_model: Default model to use for all LLM calls. Overrides
                the config defaults.
            code_model: Specific model to use for code generation. Overrides
                the default model.
            feedback_model: Specific model to use for feedback and evaluation.
                Overrides the default model.
            report_model: Specific model to use for report generation.
                Overrides the default model.
            workspace_dir: Directory for agent workspaces.
            exp_name: Name of the experiment. Random if not provided.
            log_dir: Directory for logs.
            preprocess_data: Whether to unzip archives in data directory.
            copy_data: Whether to copy data to workspace (vs symlink).
            generate_report: Whether to generate a final report.
        """
        _cfg = _load_cfg(use_cli_args=False)
        _cfg.data_dir = data_dir
        _cfg.goal = goal
        _cfg.eval = eval

        # Set default model if provided
        if default_model:
            _cfg.agent.code.model = default_model
            _cfg.agent.feedback.model = default_model
            _cfg.report.model = default_model

        # Override specific models if provided
        if code_model:
            _cfg.agent.code.model = code_model
        if feedback_model:
            _cfg.agent.feedback.model = feedback_model
        if report_model:
            _cfg.report.model = report_model

        # Set additional config parameters if provided
        if workspace_dir:
            _cfg.workspace_dir = workspace_dir
        if exp_name:
            _cfg.exp_name = exp_name
        if log_dir:
            _cfg.log_dir = log_dir
        if preprocess_data is not None:
            _cfg.preprocess_data = preprocess_data
        if copy_data is not None:
            _cfg.copy_data = copy_data
        if generate_report is not None:
            _cfg.generate_report = generate_report

        self.cfg = prep_cfg(_cfg)

        self.task_desc = load_task_desc(self.cfg)

        with Status("Preparing agent workspace..."):
            prep_agent_workspace(self.cfg)

        self.journal = Journal()
        self.agent = Agent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
        )
        self.interpreter = Interpreter(
            self.cfg.workspace_dir,
            **OmegaConf.to_container(self.cfg.exec)  # type: ignore
        )

    def run(self, steps: int) -> Solution:
        for _i in range(steps):
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.journal)
        self.interpreter.cleanup_session()

        best_node = self.journal.get_best_node(only_good=False)
        return Solution(code=best_node.code, valid_metric=best_node.metric.value)

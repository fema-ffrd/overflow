from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks.

    This protocol implements a hierarchical progress structure designed to track
    long-running hydrological operations.

    The hierarchy levels are:

    * **Phase**: High-level operation (e.g., 'Breaching paths').
    * **Step**: Named sub-operation within a phase (e.g., 'Process chunks').
    * **Message**: Detail within a step (e.g., 'Chunk 28/36').
    * **Progress**: Float (0.0-1.0) representing completion of the current step.

    Args:
        phase (str | None): The name of the high-level phase. If `None`, the
            previously set phase is preserved.
        step_name (str | None): The name of the current step within the phase.
            If `None`, the previously set step is preserved.
        step_number (int): The current step number (1-indexed). Defaults to 0.
        total_steps (int): The total number of steps expected in this phase.
            Defaults to 0.
        message (str): A detailed status message regarding the current activity.
        progress (float): The normalized progress of the current step, ranging
            from 0.0 to 1.0.

    Returns:
        None
    """

    def __call__(
        self,
        phase: str | None = None,
        step_name: str | None = None,
        step_number: int = 0,
        total_steps: int = 0,
        message: str = "",
        progress: float = 0.0,
    ) -> None:
        """Report progress for an operation."""
        ...


def silent_callback(
    phase: str | None = None,
    step_name: str | None = None,
    step_number: int = 0,
    total_steps: int = 0,
    message: str = "",
    progress: float = 0.0,
) -> None:
    """A no-op callback that does nothing.

    This is the default callback used when no progress reporting is needed.
    """
    pass


class ProgressTracker:
    """Helper class to manage progress state and emit callbacks.

    This class simplifies progress reporting by maintaining state and
    calculating progress percentages automatically using the hierarchical structure.

    Args:
        callback: The progress callback function to call with updates
        phase: The name of the phase being tracked (set once on init)
        total_steps: Total number of steps in the phase

    """

    def __init__(
        self,
        callback: ProgressCallback | None,
        phase: str,
        total_steps: int = 1,
    ) -> None:
        """Initialize the progress tracker."""
        self.callback = callback if callback is not None else silent_callback
        self.phase = phase
        self.total_steps = max(1, total_steps)
        self.current_step = 0
        # Set phase once on init
        self.callback(phase=phase)

    def update(
        self,
        step: int | None = None,
        step_name: str = "",
        message: str = "",
        progress: float | None = None,
    ) -> None:
        """Update progress and emit callback.

        Args:
            step: Current step number (if None, increments from last step)
            step_name: Name of the current step
            message: Optional detailed message
            progress: Progress within current step (0.0-1.0, defaults to 0.0)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Default to 0.0 if None, otherwise clamp to valid range
        progress = 0.0 if progress is None else max(0.0, min(1.0, progress))

        self.callback(
            step_name=step_name if step_name else None,
            step_number=self.current_step,
            total_steps=self.total_steps,
            progress=progress,
            message=message,
        )

    def step_tracker(self, step: int, total: int, message: str) -> None:
        """Report progress for a sub-step within the current step.

        Useful for reporting progress within loops or batches.

        Args:
            step: Current sub-step (e.g., current tile number)
            total: Total sub-steps (e.g., total tiles)
            message: Status message
        """
        # Calculate progress within the current step
        step_progress = step / max(1, total)

        self.callback(
            step_number=self.current_step,
            total_steps=self.total_steps,
            progress=step_progress,  # Report progress within this step, not overall
            message=message,
        )

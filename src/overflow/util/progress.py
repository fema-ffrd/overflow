"""Progress reporting utilities for overflow.

This module provides a standardized progress callback system that allows
API functions to report progress without coupling to specific display mechanisms.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks.

    Callbacks receive progress updates with the following parameters:
    - phase: High-level operation name (e.g., 'Fill Depressions')
    - step: Current step number (1-indexed)
    - total_steps: Total number of steps in the operation
    - progress: Progress as a float between 0.0 and 1.0
    - message: Detailed message about current activity
    """

    def __call__(
        self,
        phase: str,
        step: int,
        total_steps: int,
        progress: float,
        message: str,
    ) -> None:
        """Report progress for an operation."""
        ...


def silent_callback(
    phase: str,
    step: int,
    total_steps: int,
    progress: float,
    message: str,
) -> None:
    """A no-op callback that does nothing.

    This is the default callback used when no progress reporting is needed.
    """
    pass


class ProgressTracker:
    """Helper class to manage progress state and emit callbacks.

    This class simplifies progress reporting by maintaining state and
    calculating progress percentages automatically.

    Args:
        callback: The progress callback function to call with updates
        phase: The name of the operation being tracked
        total_steps: Total number of steps in the operation

    Example:
        >>> tracker = ProgressTracker(my_callback, "Processing Data", total_steps=3)
        >>> tracker.update(1, "Loading data...")
        >>> tracker.update(2, "Processing data...")
        >>> tracker.update(3, "Saving results...")
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

    def update(
        self,
        step: int | None = None,
        message: str = "",
        progress: float | None = None,
    ) -> None:
        """Update progress and emit callback.

        Args:
            step: Current step number (if None, increments from last step)
            message: Status message to display
            progress: Manual progress override (0.0-1.0, if None auto-calculates)
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if progress is None:
            # Auto-calculate progress based on step
            progress = min(1.0, self.current_step / self.total_steps)
        else:
            # Clamp manual progress to valid range
            progress = max(0.0, min(1.0, progress))

        self.callback(
            phase=self.phase,
            step=self.current_step,
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
        overall_progress = (self.current_step - 1 + step_progress) / self.total_steps

        self.callback(
            phase=self.phase,
            step=self.current_step,
            total_steps=self.total_steps,
            progress=overall_progress,
            message=message,
        )

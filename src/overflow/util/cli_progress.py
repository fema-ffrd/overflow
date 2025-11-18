"""Rich-based CLI progress display for overflow commands.

This module provides a unified progress display system that consumes
progress callbacks and renders them beautifully using Rich.
"""

from contextlib import contextmanager
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text


class PercentageColumn(ProgressColumn):
    """Renders percentage complete with color coding."""

    def render(self, task: Any) -> Text:
        """Render the percentage."""
        if task.total is None or task.total == 0:
            return Text("---%", style="progress.percentage")

        percentage = task.completed / task.total * 100
        if percentage >= 100:
            style = "green"
        elif percentage >= 50:
            style = "yellow"
        else:
            style = "cyan"

        return Text(f"{percentage:>3.0f}%", style=style)


class RichProgressDisplay:
    """Rich-based progress display that consumes progress callbacks.

    This class provides a hybrid progress display:
    - Progress bars for quantifiable operations (tiles, chunks, known steps)
    - Spinners for indeterminate operations

    The display automatically chooses the appropriate visualization based
    on the progress updates it receives.

    Args:
        console: Rich Console instance (if None, creates a new one)
        show_progress: Whether to show progress (False = silent mode)

    Example:
        >>> display = RichProgressDisplay()
        >>> with display.progress_context("Processing Data"):
        ...     # Use display.callback as the progress_callback
        ...     process_data(progress_callback=display.callback)
    """

    def __init__(
        self,
        console: Console | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the progress display."""
        self.console = console if console is not None else Console(force_terminal=True)
        self.show_progress = show_progress
        self.progress: Progress | None = None
        self.current_task: TaskID | None = None
        self.current_phase: str = ""
        self.use_bar_mode: bool = True

    def callback(
        self,
        phase: str,
        step: int,
        total_steps: int,
        progress: float,
        message: str,
    ) -> None:
        """Progress callback that updates the Rich display.

        This method is designed to be passed as the progress_callback parameter
        to overflow API functions.
        """
        if not self.show_progress or self.progress is None:
            return

        # Detect if we should use bar mode or spinner mode
        # Use bars when we have clear step progression or quantifiable progress
        # Use spinner for indeterminate or continuous operations
        if total_steps > 1 or progress > 0:
            self.use_bar_mode = True
        else:
            self.use_bar_mode = False

        # Update or create task
        if self.current_task is None or self.current_phase != phase:
            # New phase - create new task
            if self.current_task is not None:
                # Complete the previous task
                self.progress.update(self.current_task, completed=100, total=100)

            self.current_phase = phase
            if self.use_bar_mode:
                # Create task with total = 100 for percentage-based updates
                self.current_task = self.progress.add_task(
                    f"[cyan]{phase}[/cyan]",
                    total=100,
                    completed=int(progress * 100),
                )
            else:
                # Spinner mode - no total
                self.current_task = self.progress.add_task(
                    f"[cyan]{phase}[/cyan]",
                    total=None,
                )
        else:
            # Update existing task
            if self.use_bar_mode:
                # Update progress bar
                self.progress.update(
                    self.current_task,
                    completed=progress * 100,
                    description=f"[cyan]{phase}[/cyan]: {message}"
                    if message
                    else f"[cyan]{phase}[/cyan]",
                )
            else:
                # Update spinner message
                self.progress.update(
                    self.current_task,
                    description=f"[cyan]{phase}[/cyan]: {message}"
                    if message
                    else f"[cyan]{phase}[/cyan]",
                )

    @contextmanager
    def progress_context(self, initial_message: str = ""):
        """Context manager that creates a progress display for the duration of an operation.

        Args:
            initial_message: Optional initial message to display

        Example:
            >>> display = RichProgressDisplay()
            >>> with display.progress_context("Processing"):
            ...     some_function(progress_callback=display.callback)
        """
        if not self.show_progress:
            # Silent mode - just yield without creating progress display
            yield self
            return

        # Create Rich Progress with hybrid columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            PercentageColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        )

        try:
            self.progress.start()
            if initial_message:
                # Add an initial task
                self.current_task = self.progress.add_task(
                    f"[cyan]{initial_message}[/cyan]",
                    total=None,
                )
            yield self
        finally:
            # Complete any remaining tasks
            if self.current_task is not None and self.use_bar_mode:
                self.progress.update(self.current_task, completed=100, total=100)
            self.progress.stop()
            self.progress = None
            self.current_task = None
            self.current_phase = ""


def create_progress_display(
    console: Console | None = None,
    silent: bool = False,
) -> RichProgressDisplay:
    """Factory function to create a progress display.

    Args:
        console: Rich Console instance (if None, creates a new one)
        silent: If True, creates a silent display that shows nothing

    Returns:
        RichProgressDisplay instance
    """
    return RichProgressDisplay(console=console, show_progress=not silent)

import re
import sys
import time
from contextlib import contextmanager

from rich.console import Console


def format_duration(seconds: float) -> str:
    """Convert seconds into compact string like 1h2m32s.

    Args:
        seconds: Duration in seconds

    Returns:
        Compact duration string
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []

    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return "".join(parts)


class RichProgressDisplay:
    def __init__(
        self,
        console: Console | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the progress display."""
        self.console = console if console is not None else Console(force_terminal=True)
        self.show_progress = show_progress
        self.current_phase: str = ""
        self.current_step: str = ""
        self.step_start_time: float = 0.0
        self.in_chunk_progress: bool = False
        self.step_timing_printed: bool = False

    def callback(
        self,
        phase: str | None = None,
        step_name: str | None = None,
        step_number: int = 0,
        total_steps: int = 0,
        message: str = "",
        progress: float = 0.0,
    ) -> None:
        """Progress callback."""
        if not self.show_progress:
            return

        # Print new phase
        if phase is not None and phase != self.current_phase:
            # If we were showing chunk progress, print newline to finish that line
            if self.in_chunk_progress:
                print()
                self.in_chunk_progress = False

            self.current_phase = phase
            self.console.print(f"\n[bold cyan]{phase}[/bold cyan]")
            self.current_step = ""

        # Handle step change
        if step_name is not None:
            if total_steps > 1:
                new_step = f"{step_number}/{total_steps} {step_name}"
            else:
                new_step = step_name

            if new_step != self.current_step:
                # Finish previous step (print final line with timing)
                if self.current_step and not self.step_timing_printed:
                    elapsed = time.time() - self.step_start_time
                    # Clear the line and print final step line with timing
                    sys.stdout.write("\r\033[K")
                    print(f"  {self.current_step} ({format_duration(elapsed)})")
                    self.step_timing_printed = True
                    self.in_chunk_progress = False

                # Start new step
                self.current_step = new_step
                self.step_start_time = time.time()
                self.step_timing_printed = False
                print(f"  {self.current_step}", end="", flush=True)

        # Handle chunk progress
        if message and re.match(r"Chunk\s+\d+/\d+", message):
            match = re.match(r"Chunk\s+(\d+)/(\d+)", message)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                percentage = int((current / total) * 100)

                # Show live chunk progress
                sys.stdout.write(
                    f"\r\033[K  {self.current_step} {current}/{total} ({percentage}%)"
                )
                sys.stdout.flush()
                self.in_chunk_progress = True

                # If this is the last chunk, finish the step with timing
                if current == total:
                    elapsed = time.time() - self.step_start_time
                    # Clear the chunk progress line and print final step line with timing
                    sys.stdout.write("\r\033[K")
                    print(f"  {self.current_step} ({format_duration(elapsed)})")
                    self.in_chunk_progress = False
                    self.step_timing_printed = True

    @contextmanager
    def progress_context(self, initial_message: str = ""):
        """Context manager for progress display."""
        if not self.show_progress:
            yield self
            return

        try:
            if initial_message:
                self.current_phase = initial_message
                self.console.print(f"\n[bold cyan]{initial_message}[/bold cyan]")
            yield self
        finally:
            # Finish any remaining step
            if self.current_step and not self.step_timing_printed:
                elapsed = time.time() - self.step_start_time
                # Clear the line and print final step line with timing
                if elapsed > 0:  # Only if step actually ran
                    sys.stdout.write("\r\033[K")
                    print(f"  {self.current_step} ({format_duration(elapsed)})")

            self.current_phase = ""
            self.current_step = ""
            self.step_timing_printed = False
            self.in_chunk_progress = False


def create_progress_display(
    console: Console | None = None,
    silent: bool = False,
) -> RichProgressDisplay:
    """Factory function to create a progress display."""
    return RichProgressDisplay(console=console, show_progress=not silent)

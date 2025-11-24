import re
import shutil
import sys
import threading
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
    minutes, secs_int = divmod(remainder, 60)
    parts = []

    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")

    # For durations under 1 minute, show decimal seconds for precision
    if hours == 0 and minutes == 0:
        parts.append(f"{seconds:.1f}s")
    else:
        parts.append(f"{secs_int}s")

    return "".join(parts)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text.

    Args:
        text: Text potentially containing ANSI codes

    Returns:
        Text with ANSI codes removed
    """
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def format_step_line(left_text: str, right_text: str, min_dashes: int = 3) -> str:
    """Format a line with left-aligned text and right-aligned text with dashes.

    Args:
        left_text: Text to show on the left
        right_text: Text to show on the right
        min_dashes: Minimum number of dashes to show

    Returns:
        Formatted line with dashes filling the space
    """
    terminal_width = shutil.get_terminal_size().columns
    # Account for spacing: "  " at start, " " before right text
    available_width = terminal_width - 3

    # Calculate dash count using visible text length (strip ANSI codes)
    visible_left = strip_ansi_codes(left_text)
    visible_right = strip_ansi_codes(right_text)
    text_width = len(visible_left) + len(visible_right)
    dash_count = max(min_dashes, available_width - text_width)

    return f"  {left_text} {'-' * dash_count} {right_text}"


class RichProgressDisplay:
    # Custom spinner frames
    SPINNER_FRAMES = ["■", "≡", "=", "-", "=", "≡"]
    # Max step name width for alignment
    MAX_STEP_WIDTH = 34

    def __init__(
        self,
        console: Console | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the progress display."""
        self.console = console if console is not None else Console(force_terminal=True)
        self.show_progress = show_progress
        self.is_tty = sys.stdout.isatty()  # Detect if output is to a terminal
        self.current_phase: str = ""
        self.current_step: str = ""
        self.step_start_time: float = 0.0
        self.in_chunk_progress: bool = False
        self.step_timing_printed: bool = False
        self.spinner_index: int = 0
        self.spinner_thread: threading.Thread | None = None
        self.spinner_stop_event: threading.Event = threading.Event()
        self.spinner_lock: threading.Lock = threading.Lock()
        self.chunk_info: str = ""  # Store current chunk info for spinner updates
        self.last_logged_percentage: int = (
            -1
        )  # Track last logged percentage for non-TTY mode

    def _spinner_worker(self) -> None:
        """Background thread that updates the spinner continuously."""
        while not self.spinner_stop_event.is_set():
            if self.current_step and not self.step_timing_printed:
                with self.spinner_lock:
                    # Advance spinner
                    self.spinner_index = (self.spinner_index + 1) % len(
                        self.SPINNER_FRAMES
                    )
                    spinner = self.SPINNER_FRAMES[self.spinner_index]

                    # Update display with bold cyan spinner (matching phase color)
                    if self.chunk_info:
                        # Parse chunk info to extract percentage and counts
                        # Format: "X/Y (ZZ%)" -> show as "■ | ZZ%| ■■■■------ | time | X/Y Step name"
                        match = re.match(r"(\d+)/(\d+)\s+\((\d+)%\)", self.chunk_info)
                        if match:
                            current = int(match.group(1))
                            total = int(match.group(2))
                            counts = f"{current}/{total}"
                            percentage_str = match.group(3).rjust(
                                3
                            )  # Right-align to 3 chars
                            percentage_val = int(match.group(3))

                            # Calculate elapsed time
                            elapsed = time.time() - self.step_start_time
                            time_info = format_duration(elapsed)

                            # Build 10-character progress bar
                            bar_length = 10
                            filled_count = int(
                                percentage_val / 10
                            )  # Each position = 10%

                            # Calculate which frame to show for the growing edge based on sub-percentage
                            sub_percentage = (
                                percentage_val % 10
                            )  # 0-9 within current position
                            # Map 0-9 to frame: 0-3: -, 4-6: =, 7-9: ≡
                            # SPINNER_FRAMES = ["■", "≡", "=", "-", "=", "≡"]
                            # "-" is index 3, "=" is index 2, "≡" is index 1
                            frame_map = [
                                3,
                                3,
                                3,
                                3,
                                2,
                                2,
                                2,
                                1,
                                1,
                                1,
                            ]  # 0-33%: -, 34-66%: =, 67-99%: ≡
                            frame_idx = frame_map[sub_percentage]
                            growing_edge = self.SPINNER_FRAMES[frame_idx]

                            # Build progress bar with animation
                            progress_bar = ""
                            for i in range(bar_length):
                                if i < filled_count:
                                    progress_bar += "■"
                                elif i == filled_count and percentage_val < 100:
                                    # Show growing edge with appropriate frame
                                    progress_bar += growing_edge
                                else:
                                    progress_bar += "-"

                            sys.stdout.write(
                                f"\r\033[K  \033[1;36m{spinner}\033[0m |{percentage_str}%| \033[36m{progress_bar}\033[0m {counts} | {time_info} | {self.current_step}"
                            )
                        else:
                            sys.stdout.write(
                                f"\r\033[K  \033[1;36m{spinner}\033[0m {self.current_step} {self.chunk_info}"
                            )
                    else:
                        # Show step without chunk info
                        sys.stdout.write(
                            f"\r\033[K  \033[1;36m{spinner}\033[0m {self.current_step}"
                        )
                    sys.stdout.flush()

            # Update every 0.1 seconds
            self.spinner_stop_event.wait(0.1)

    def _start_spinner(self) -> None:
        """Start the spinner animation thread."""
        self._stop_spinner()  # Stop any existing thread
        self.spinner_stop_event.clear()
        self.spinner_thread = threading.Thread(target=self._spinner_worker, daemon=True)
        self.spinner_thread.start()

    def _stop_spinner(self) -> None:
        """Stop the spinner animation thread."""
        if self.spinner_thread and self.spinner_thread.is_alive():
            self.spinner_stop_event.set()
            self.spinner_thread.join(timeout=0.5)
            self.spinner_thread = None

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

        # Non-TTY mode: simple text output without spinners or colors
        if not self.is_tty:
            # Print new phase
            if phase is not None and phase != self.current_phase:
                self.current_phase = phase
                print(f"\n{phase}", flush=True)
                self.current_step = ""
                self.last_logged_percentage = -1

            # Handle step change
            if step_name is not None:
                if total_steps > 1:
                    new_step = f"{step_number}/{total_steps} {step_name}"
                else:
                    new_step = step_name

                if new_step != self.current_step:
                    # Finish previous step with timing
                    if self.current_step:
                        elapsed = time.time() - self.step_start_time
                        print(
                            f"  {self.current_step} - {format_duration(elapsed)}",
                            flush=True,
                        )

                    self.current_step = new_step
                    self.step_start_time = time.time()
                    self.step_timing_printed = False
                    self.last_logged_percentage = -1

            # Handle chunk progress - log every increment
            if message and re.match(r"Chunk\s+\d+/\d+", message):
                match = re.match(r"Chunk\s+(\d+)/(\d+)", message)
                if match:
                    current = int(match.group(1))
                    total = int(match.group(2))
                    percentage = int((current / total) * 100)

                    # Log every chunk
                    print(
                        f"  {self.current_step}: {percentage}% ({current}/{total})",
                        flush=True,
                    )

                    # Mark step as timing printed when complete
                    if current == total:
                        self.step_timing_printed = True
            return

        # TTY mode: full interactive display with spinners and colors
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
                    self._stop_spinner()
                    elapsed = time.time() - self.step_start_time
                    # Clear the line and print final step line with timing
                    sys.stdout.write("\r\033[K")
                    padded_step = self.current_step.ljust(self.MAX_STEP_WIDTH)
                    print(f"  {padded_step} ({format_duration(elapsed)})")
                    self.step_timing_printed = True
                    self.in_chunk_progress = False

                # Start new step with spinner
                with self.spinner_lock:
                    self.current_step = new_step
                    self.step_start_time = time.time()
                    self.step_timing_printed = False
                    self.spinner_index = 0
                    self.chunk_info = ""
                self._start_spinner()

        # Handle chunk progress
        if message and re.match(r"Chunk\s+\d+/\d+", message):
            match = re.match(r"Chunk\s+(\d+)/(\d+)", message)
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                percentage = int((current / total) * 100)

                # Update chunk info for spinner thread to display
                with self.spinner_lock:
                    self.chunk_info = f"{current}/{total} ({percentage}%)"
                    self.in_chunk_progress = True

                # If this is the last chunk, finish the step with timing
                if current == total:
                    self._stop_spinner()
                    elapsed = time.time() - self.step_start_time
                    # Clear the chunk progress line and print final step line with timing
                    sys.stdout.write("\r\033[K")
                    padded_step = self.current_step.ljust(self.MAX_STEP_WIDTH)
                    print(f"  {padded_step} ({format_duration(elapsed)})")
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
                if self.is_tty:
                    self.console.print(f"\n[bold cyan]{initial_message}[/bold cyan]")
                else:
                    print(f"\n{initial_message}", flush=True)
            yield self
        finally:
            # Finish any remaining step
            if self.current_step and not self.step_timing_printed:
                elapsed = time.time() - self.step_start_time
                if elapsed > 0:  # Only if step actually ran
                    if self.is_tty:
                        # TTY mode: clear line and show aligned output
                        self._stop_spinner()
                        sys.stdout.write("\r\033[K")
                        padded_step = self.current_step.ljust(self.MAX_STEP_WIDTH)
                        print(f"  {padded_step} ({format_duration(elapsed)})")
                    else:
                        # Non-TTY mode: simple text output
                        print(
                            f"  {self.current_step} - {format_duration(elapsed)}",
                            flush=True,
                        )

            # Ensure spinner is stopped
            self._stop_spinner()

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

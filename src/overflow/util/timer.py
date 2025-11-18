"""Timing and resource tracking utilities for overflow commands."""

import os
import time
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

console = Console(force_terminal=True)


class ResourceStats:
    """Track timing and output file information for operations.

    This class maintains statistics about operations performed during
    command execution, including timing and output files generated.
    """

    def __init__(self) -> None:
        """Initialize the resource stats tracker."""
        self.stats: dict[str, float] = {}
        self.operation_order: list[str] = []
        self.output_files: list[tuple[str, Path]] = []

    def add_stats(self, description: str, duration: float) -> None:
        """Add timing statistics for an operation.

        Args:
            description: Name of the operation
            duration: Duration in seconds
        """
        if description not in self.stats and description != "Total processing":
            self.operation_order.append(description)
        self.stats[description] = duration

    def add_output_file(self, description: str, file_path: Path | str) -> None:
        """Add an output file to the tracking.

        Args:
            description: Description of the file (e.g., "Flow Direction")
            file_path: Path to the output file
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        self.output_files.append((description, file_path))

    @staticmethod
    def format_compact_duration(seconds: float) -> str:
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

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Convert bytes to human readable string.

        Args:
            size_bytes: File size in bytes

        Returns:
            Human-readable file size
        """
        size: float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} PB"

    def get_timing_table(self) -> Table:
        """Create a Rich Table showing operation timing breakdown.

        Returns:
            Rich Table with timing information
        """
        table = Table(
            title="Operation Timing", show_header=True, header_style="bold cyan"
        )
        table.add_column("Operation", style="blue", no_wrap=False)
        table.add_column("Duration", style="cyan", justify="right")
        table.add_column("Timeline", style="bright_blue")

        if not self.stats:
            return table

        # Filter out total processing from the chart
        chart_data = {k: v for k, v in self.stats.items() if k != "Total processing"}

        if not chart_data:
            return table

        # Sort by operation order
        sorted_items = [
            (op, chart_data[op]) for op in self.operation_order if op in chart_data
        ]

        max_duration = max(v for v in chart_data.values())
        chart_width = 30

        for label, duration in sorted_items:
            time_bar_length = (
                int((duration / max_duration) * chart_width) if max_duration > 0 else 0
            )
            table.add_row(
                label,
                self.format_compact_duration(duration),
                "█" * time_bar_length,
            )

        return table

    def get_output_files_table(self) -> Table | None:
        """Create a Rich Table showing output files.

        Returns:
            Rich Table with file information, or None if no files
        """
        if not self.output_files:
            return None

        table = Table(title="Output Files", show_header=True, header_style="bold cyan")
        table.add_column("Type", style="blue")
        table.add_column("Path", style="cyan", no_wrap=False)
        table.add_column("Size", style="green", justify="right")

        for description, file_path in self.output_files:
            if file_path.exists():
                size = os.path.getsize(file_path)
                table.add_row(
                    description,
                    str(file_path),
                    self.format_file_size(size),
                )
            else:
                table.add_row(description, str(file_path), "[red]Not found[/red]")

        return table

    def get_summary_panel(self, success: bool = True) -> Panel:
        """Create a comprehensive summary panel.

        Args:
            success: Whether the operation completed successfully

        Returns:
            Rich Panel with complete summary
        """
        renderables: list = []

        # Status
        if success:
            renderables.append(
                Text("✓ Operation completed successfully", style="bold green")
            )
        else:
            renderables.append(Text("✗ Operation failed", style="bold red"))
        renderables.append(Text())

        # Total time
        if "Total processing" in self.stats:
            total_duration = self.stats["Total processing"]
            renderables.append(
                Text.assemble(
                    ("Total time: ", "bold yellow"),
                    (self.format_compact_duration(total_duration), "cyan"),
                )
            )
            renderables.append(Text())

        # Timing breakdown table
        if self.stats:
            timing_table = self.get_timing_table()
            renderables.append(timing_table)
            renderables.append(Text())

        # Output files table
        output_table = self.get_output_files_table()
        if output_table:
            renderables.append(output_table)

        return Panel(
            Group(*renderables),
            title="[bold blue]Summary[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )


# Global resource stats instance
resource_stats = ResourceStats()


class Timer:
    """Timer utility for formatting durations."""

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Convert seconds into human readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Human-readable duration (e.g., "2 hours, 3 minutes and 45 seconds")
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        parts = []

        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if secs > 0 or not parts:
            parts.append(f"{secs} second{'s' if secs != 1 else ''}")

        if len(parts) > 1:
            return f"{', '.join(parts[:-1])} and {parts[-1]}"
        return parts[0]


@contextmanager
def timer(
    description: str,
    silent: bool = False,
    spinner: bool = False,
):
    """Context manager for timing code blocks with rich output.

    Args:
        description: Description of the operation being timed
        silent: If True, suppresses all output
        spinner: If True, shows a spinner during execution

    Example:
        >>> with timer("Processing data", spinner=True):
        ...     process_data()
        ✓ Processing data completed in 2 minutes and 34 seconds
    """
    start_time = time.time()

    if spinner and not silent:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description, total=None)
            try:
                yield
            finally:
                duration = time.time() - start_time
                resource_stats.add_stats(description, duration)
                progress.stop()

                console.print(
                    f"[green]✓[/green] {description} completed in "
                    f"[bold cyan]{Timer.format_duration(duration)}[/bold cyan]"
                )
    else:
        if not silent:
            console.print(f"[bold blue]{description}...[/bold blue]")
        try:
            yield
        finally:
            duration = time.time() - start_time
            resource_stats.add_stats(description, duration)

            if not silent:
                console.print(
                    f"[green]✓[/green] {description} completed in "
                    f"[bold cyan]{Timer.format_duration(duration)}[/bold cyan]"
                )

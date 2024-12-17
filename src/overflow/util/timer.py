from contextlib import contextmanager
import time
import psutil
import threading
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


class MemoryMonitor:
    def __init__(self):
        self.keep_monitoring = True
        self.max_memory = 0
        self.process = psutil.Process()

    def monitor(self):
        while self.keep_monitoring:
            current_memory = self.process.memory_info().rss
            self.max_memory = max(self.max_memory, current_memory)
            time.sleep(0.1)  # Check every 100ms

    def start(self):
        self.keep_monitoring = True
        self.thread = threading.Thread(target=self.monitor)
        self.thread.start()

    def stop(self):
        self.keep_monitoring = False
        self.thread.join()
        return self.max_memory


class ResourceStats:
    def __init__(self):
        self.stats = {}
        self.operation_order = []

    def add_stats(self, description: str, duration: float, memory: float):
        if description not in self.stats and description != "Total processing":
            self.operation_order.append(description)
        self.stats[description] = (duration, memory)

    @staticmethod
    def format_compact_duration(seconds: float) -> str:
        """Convert seconds into compact string like 1h2m32s"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []

        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")

        return "".join(parts)

    @staticmethod
    def format_memory(bytes_value: float) -> str:
        """Convert bytes to human readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_value < 1024:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024
        return f"{bytes_value:.1f}PB"

    def get_chart(self) -> Panel:
        if not self.stats:
            return Panel("No stats available")

        chart_data = {k: v for k, v in self.stats.items() if k != "Total processing"}

        sorted_items = [
            (op, chart_data[op]) for op in self.operation_order if op in chart_data
        ]

        output = []
        output.append(Text("\nOperation Timing and Memory Usage", style="bold cyan"))
        output.append("")

        max_label_len = max(len(item[0]) for item in sorted_items)
        max_label_len = min(max_label_len, 25)

        headers = Text()
        headers.append(f"{'Operation':<{max_label_len}} ", style="bold blue")
        headers.append(f"{'Time':<10} ", style="bold blue")
        headers.append(f"{'Memory':<10} ", style="bold blue")
        headers.append("Usage", style="bold blue")
        output.append(headers)
        output.append("")

        max_duration = max(v[0] for v in chart_data.values())
        chart_width = 30

        for label, (duration, memory) in sorted_items:
            display_label = label[:max_label_len] + (
                "..." if len(label) > max_label_len else ""
            )

            time_bar_length = (
                int((duration / max_duration) * chart_width) if max_duration > 0 else 0
            )

            line = Text()
            line.append(f"{display_label:<{max_label_len}} ", style="blue")
            line.append(f"{self.format_compact_duration(duration):<10} ", style="cyan")
            line.append(f"{self.format_memory(memory):<10} ", style="green")
            line.append("█" * time_bar_length, style="bright_blue")

            output.append(line)

        if "Total processing" in self.stats:
            total_duration = self.stats["Total processing"][0]
            output.append("")
            total_line = Text()
            total_line.append(
                f"Total time: {self.format_compact_duration(total_duration)}",
                style="bold yellow",
            )
            output.append(total_line)

        return Panel(
            "\n".join([str(line) for line in output]),
            title="Resource Usage Summary",
            border_style="blue",
        )


resource_stats = ResourceStats()


class Timer:
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Convert seconds into human readable string."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        parts = []

        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        if len(parts) > 1:
            return f"{', '.join(parts[:-1])} and {parts[-1]}"
        return parts[0]


@contextmanager
def timer(
    description: str,
    silent: bool = False,
    spinner: bool = False,
    track_memory: bool = True,
):
    """
    Context manager for timing code blocks with rich output.

    Parameters
    ----------
    description : str
        Description of the operation being timed
    silent : bool
        If True, suppresses all output
    spinner : bool
        If True, shows a spinner during execution
    track_memory : bool
        If True, tracks memory usage
    """
    start_time = time.time()

    # Initialize memory monitoring if requested
    if track_memory:
        monitor = MemoryMonitor()
        monitor.start()

    if spinner and not silent:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(description, total=None)
            try:
                yield
            finally:
                duration = time.time() - start_time

                if track_memory:
                    memory_used = monitor.stop()
                else:
                    memory_used = 0

                resource_stats.add_stats(description, duration, memory_used)
                progress.stop()

                if track_memory:
                    console.print(
                        f"[green]✓[/green] {description} completed in "
                        f"[bold cyan]{Timer.format_duration(duration)}[/bold cyan] | "
                        f"Memory used: [bold green]{ResourceStats.format_memory(memory_used)}[/bold green]"
                    )
                else:
                    console.print(
                        f"[green]✓[/green] {description} completed in "
                        f"[bold cyan]{Timer.format_duration(duration)}[/bold cyan]"
                    )
    else:
        if not silent:
            console.print(f"[bold blue]{description}...[/bold blue]", end="\n")
        try:
            yield
        finally:
            duration = time.time() - start_time

            if track_memory:
                memory_used = monitor.stop()
            else:
                memory_used = 0

            resource_stats.add_stats(description, duration, memory_used)

            if not silent:
                if track_memory:
                    console.print(
                        f"[green]✓[/green] {description} completed in "
                        f"[bold cyan]{Timer.format_duration(duration)}[/bold cyan] | "
                        f"Memory used: [bold green]{ResourceStats.format_memory(memory_used)}[/bold green]"
                    )
                else:
                    console.print(
                        f"[green]✓[/green] {description} completed in "
                        f"[bold cyan]{Timer.format_duration(duration)}[/bold cyan]"
                    )

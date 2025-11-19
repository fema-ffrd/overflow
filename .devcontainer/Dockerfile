FROM ghcr.io/osgeo/gdal:ubuntu-small-3.12.0

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

RUN uv venv $UV_PROJECT_ENVIRONMENT --system-site-packages

ARG USERNAME=ubuntu
RUN apt-get update \
    && apt-get install -y \
    sudo \
    git \
    build-essential \
    curl \
    wget \
    ripgrep \
    fd-find \
    jq \
    zip \
    unzip \
    fish \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/$USERNAME/.config/fish \
    && mkdir -p /home/$USERNAME/.config \
    && mkdir -p /commandhistory \
    && touch /commandhistory/fish_history

RUN printf '\
# Disable welcome message\n\
set -g fish_greeting\n\
\n\
# Set the history file to a custom location\n\
set -gx HISTFILE /commandhistory/fish_history\n\
\n\
# Initialize Starship prompt\n\
starship init fish | source\n\
' > /home/$USERNAME/.config/fish/config.fish

RUN chsh -s /usr/bin/fish $USERNAME
RUN curl -sS https://starship.rs/install.sh | sh -s -- -y

RUN printf '\
format = """\\\n\
$directory\\\n\
$git_branch\\\n\
$git_status\\\n\
$python\\\n\
${custom.gdal}\\\n\
$line_break\\\n\
$character"""\n\
\n\
[directory]\n\
style = "blue"\n\
truncation_length = 3\n\
truncate_to_repo = false\n\
read_only = " 🔒"\n\
\n\
[character]\n\
success_symbol = "[❯](purple)"\n\
error_symbol = "[❯](red)"\n\
vimcmd_symbol = "[❮](green)"\n\
\n\
[git_branch]\n\
symbol = "🌱 "\n\
style = "green"\n\
\n\
[git_status]\n\
format = "[[(*$conflicted$untracked$modified$staged$renamed$deleted)](218) ($ahead_behind$stashed)]($style)"\n\
style = "cyan"\n\
\n\
[python]\n\
symbol = "🐍 "\n\
format = "[${symbol}${version}](yellow) "\n\
\n\
[custom.gdal]\n\
command = "gdal-config --version"\n\
when = "command -v gdal-config"\n\
symbol = "🌎 "\n\
format = "[$symbol($output )](blue) "\n\
' > /home/$USERNAME/.config/starship.toml

RUN chown -R $USERNAME:$USERNAME /commandhistory \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME \
    && chown -R $USERNAME:$USERNAME $UV_PROJECT_ENVIRONMENT

USER $USERNAME

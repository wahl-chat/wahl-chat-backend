# SPDX-FileCopyrightText: 2025 2025 wahl.chat
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

FROM python:3.11.3-slim

# Install Poetry
RUN pip install poetry

# Copy only the necessary files for Poetry
COPY pyproject.toml poetry.lock /app/

WORKDIR /app

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application code
COPY . /app/

CMD ["poetry", "run", "python", "-m", "src.aiohttp_app", "--host", "0.0.0.0", "--port", "8080"]

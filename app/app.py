"""Lệnh tương thích để khởi động backend demo cục bộ."""

from __future__ import annotations


def main() -> None:
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

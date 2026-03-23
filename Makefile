.PHONY: build-clews bash-clews build-discogs-vinet bash-discogs-vinet build-retrival bash-retrival

build-clews:
	docker compose build clews

bash-clews:
	docker compose run --rm clews bash

build-discogs-vinet:
	docker compose build discogs-vinet

bash-discogs-vinet:
	docker compose run --rm discogs-vinet bash

build-retrival:
	docker compose build retrival --no-cache

bash-retrival:
	docker compose run --rm retrival bash

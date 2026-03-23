.PHONY: build-clews bash-clews build-discogs-vinet bash-discogs-vinet build-retrieval bash-retrieval build-degradation bash-degradation

build-degradation:
	docker compose build degradation

bash-degradation:
	docker compose run --rm degradation bash

build-clews:
	docker compose build clews

bash-clews:
	docker compose run --rm clews bash

build-discogs-vinet:
	docker compose build discogs-vinet

bash-discogs-vinet:
	docker compose run --rm discogs-vinet bash

build-retrieval:
	docker compose build retrieval --no-cache

bash-retrieval:
	docker compose run --rm retrieval bash

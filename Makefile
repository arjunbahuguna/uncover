.PHONY: build-discogs-vinet bash-discogs-vinet

build-discogs-vinet:
	docker compose build discogs-vinet

bash-discogs-vinet:
	docker compose run --rm discogs-vinet bash

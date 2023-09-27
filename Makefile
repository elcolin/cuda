all: reload

start: fclean reload

stop:
	docker-compose -f docker-compose.yml down

clean: stop
	docker system prune -af

fclean: clean
	docker volume rm -f cuda

reload:
	docker-compose -f docker-compose.yml up -d --build

re: fclean all
.PHONY: domain stop clean reload fclean all
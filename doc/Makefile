all: queue_time_predictions.png

%.png: %.dot
	dot -Tpng $< -o $@

.PHONY: clean
clean:
	rm -f *.png

.PHONY: all figures pdf clean

MAIN = main
BIB = references.bib

all: pdf

figures:
	python generate_figures.py

pdf: figures $(MAIN).tex $(BIB)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN)
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf

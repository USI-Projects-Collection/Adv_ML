filename=main

pdf:
	pdflatex ${filename} || exit 1
	pdflatex ${filename} || exit 1

read:
	evince ${filename}.pdf &

clean:
	rm -f \
	${filename}.out \
	${filename}.log \
	${filename}.bbl \
	${filename}.blg \
	${filename}.au \
	${filename}.log \
	${filename}.ps \
	${filename}.aux \
	${filename}.out \
	${filename}.dvi \
	${filename}.bbl \
	${filename}.blg \
	${filename}.toc \
	${filename}.nav \
	${filename}.vrb \
	${filename}.snm \
	${filename}.fdb_latexmk \
	${filename}.fls \
	${filename}.synctex.gz 
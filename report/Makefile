compile:
	pandoc report.md -o report.pdf --from markdown  \
		--template eisvogel --listings --number-sections  \
		--standalone  \
		--toc=true --toc-depth=4  \
		--lof=true --lot=true \
		--filter pandoc-crossref \
		--filter pandoc-latex-environment \
		-V top-level-division=section

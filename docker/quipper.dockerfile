FROM haskell:8.8.4-buster

# Install quipper via cabal
RUN cabal update && cabal install quipper

# Make cabal's bin dir visible
ENV PATH="/root/.cabal/bin:${PATH}"

WORKDIR /work

# Default command; you can override with `docker run ... bash -lc "..."`.
CMD ["bash"]

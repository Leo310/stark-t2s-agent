"""Materialization of STaRK-Prime into SQL and RDF stores."""

from stark_prime_t2s.materialize.postgres_prime import PostgresPrimeStore
from stark_prime_t2s.materialize.fuseki_prime import FusekiPrimeStore

__all__ = [
    "PostgresPrimeStore",
    "FusekiPrimeStore",
]

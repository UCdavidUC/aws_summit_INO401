"""
Utilidad de certificados TLS para el scraping de CNBV y Banxico.

Ambos sitios (www.cnbv.gob.mx y www.banxico.org.mx) presentan cadenas de
certificados incompletas: el servidor no envia el certificado intermedio de
su CA (GlobalSign para CNBV, GoDaddy para Banxico). Navegadores y `curl`
(via macOS/Secure Transport) toleran esto completando la cadena via AIA
(Authority Information Access), pero el modulo `ssl` de Python no lo hace,
lo que provoca `CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate`.

Este modulo construye, en tiempo de ejecucion, un bundle CA combinando el
bundle de `certifi` con los certificados intermedios faltantes (guardados en
`data_pipeline/certs/`), y devuelve la ruta a ese bundle combinado para
pasarla como `verify=` en las llamadas de `requests`.
"""
import os
import tempfile

import certifi

_CERTS_DIR = os.path.join(os.path.dirname(__file__), "..", "certs")

_EXTRA_INTERMEDIATES = [
    "intermediate_globalsign_gsrsaovsslca2018.pem",  # CNBV (*.cnbv.gob.mx)
    "intermediate_godaddy_gdig2.pem",  # Banxico (www.banxico.org.mx)
]

_combined_path_cache = None


def get_ca_bundle_path() -> str:
    """Devuelve la ruta a un bundle CA = certifi + intermedios faltantes de
    CNBV/Banxico. El archivo combinado se genera una sola vez por proceso."""
    global _combined_path_cache
    if _combined_path_cache and os.path.exists(_combined_path_cache):
        return _combined_path_cache

    base_bundle = certifi.where()
    with open(base_bundle, "r", encoding="utf-8") as f:
        combined = f.read()

    for filename in _EXTRA_INTERMEDIATES:
        path = os.path.join(_CERTS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                combined += "\n" + f.read()

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pem", prefix="cnbv_banxico_ca_bundle_", delete=False
    )
    tmp.write(combined)
    tmp.close()

    _combined_path_cache = tmp.name
    return _combined_path_cache

"""Estilo de bibliografía en español para el libro.

Parte del estilo "plain" (ordenado por autor) y traduce lo que queda en inglés:
"and" -> "y", "and others" -> "et al.", "In" -> "En", "pages" -> "págs." y la edición.
"""
from collections import Counter

from pybtex.plugin import register_plugin
from pybtex.style.labels import BaseLabelStyle
from pybtex.style.sorting import BaseSortingStyle
from pybtex.richtext import Text
from pybtex.style.formatting import toplevel
from pybtex.style.formatting.plain import Style as Plain
from pybtex.style.formatting.unsrt import date, pages
from pybtex.style.template import (FieldIsMissing, field, first_of, join, node,
                                   optional, optional_field, sentence, tag, words)


@node
def nombres(children, context, role):
    """Autores separados por comas y 'y'; 'others' se escribe 'et al.'"""
    try:
        personas = context["entry"].persons[role]
    except KeyError:
        raise FieldIsMissing(role, context["entry"])
    estilo = context["style"]
    otros = len(personas) > 1 and str(personas[-1]) == "others"
    if otros:
        personas = personas[:-1]
    lista = [estilo.format_name(p, estilo.abbreviate_names) for p in personas]
    if otros:
        return Text(join(sep=", ")[lista].format_data(context), " et al.")
    return join(sep=", ", sep2=" y ", last_sep=" y ")[lista].format_data(context)


def apellido(persona):
    """Apellido completo (con 'van', 'de', etc.) y sin llaves."""
    texto = " ".join(persona.prelast_names + persona.last_names) or str(persona)
    return texto.replace("{", "").replace("}", "")


class OrdenES(BaseSortingStyle):
    """Orden alfabético por apellido del primer autor y luego por año."""

    def sorting_key(self, e):
        personas = e.persons.get("author", [])
        primero = apellido(personas[0]).lower() if personas else ""
        return (primero, e.fields.get("year", ""), e.fields.get("title", "").lower())


class EtiquetasES(BaseLabelStyle):
    """Etiquetas tipo [Apellido, año]; si se repiten, se agrega a, b, c..."""

    def etiqueta(self, e):
        personas = e.persons.get("author", [])
        apellidos = [apellido(p) for p in personas if str(p) != "others"]
        if not apellidos:
            autor = "Anónimo"
        elif len(apellidos) == 1 and len(personas) == 1:
            autor = apellidos[0]
        elif len(apellidos) == 2 and len(personas) == 2:
            autor = f"{apellidos[0]} y {apellidos[1]}"
        else:
            autor = f"{apellidos[0]} et al."
        return f"{autor}, {e.fields.get('year', 's. f.')}"

    def format_labels(self, sorted_entries):
        entradas = list(sorted_entries)
        base = [self.etiqueta(e) for e in entradas]
        repetidas = Counter(base)
        vistos = Counter()
        for b in base:
            if repetidas[b] > 1:
                vistos[b] += 1
                yield b + "abcdefghijklmnopqrstuvwxyz"[vistos[b] - 1]
            else:
                yield b


class EstiloES(Plain):
    default_label_style = "es"
    default_sorting_style = "es"

    def format_names(self, role, as_sentence=True):
        n = nombres(role=role)
        return sentence[n] if as_sentence else n

    def format_edition(self, e):
        return optional[join[field("edition"), ".ª ed."]]

    def get_article_template(self, e):
        volumen_y_paginas = first_of[
            optional[join[field("volume"), optional["(", field("number"), ")"], ":", pages]],
            words["págs.", pages],
        ]
        return toplevel[
            self.format_names("author"),
            self.format_title(e, "title"),
            sentence[tag("em")[field("journal")], optional[volumen_y_paginas], date],
            sentence[optional_field("note")],
            self.format_web_refs(e),
        ]

    def get_inproceedings_template(self, e):
        return toplevel[
            sentence[self.format_names("author")],
            self.format_title(e, "title"),
            words[
                "En",
                sentence[
                    self.format_btitle(e, "booktitle", as_sentence=False),
                    optional[words["págs.", pages]],
                ],
                self.format_address_organization_publisher_date(e),
            ],
            sentence[optional_field("note")],
            self.format_web_refs(e),
        ]


def setup(app):
    register_plugin("pybtex.style.sorting", "es", OrdenES)
    register_plugin("pybtex.style.labels", "es", EtiquetasES)
    register_plugin("pybtex.style.formatting", "es", EstiloES)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
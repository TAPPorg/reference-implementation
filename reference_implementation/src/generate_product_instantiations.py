#!/usr/bin/env python3
"""
Generates product_instantiations.gen.c: one TAPP_EXECUTE_PRODUCT_TEMPLATE
instantiation per (type_A, type_B, type_C, type_D) combination, plus the
TAPP_PRODUCT_DISPATCH table used by TAPP_execute_product to pick the right
one at runtime.

Regenerate after changing the set of supported TAPP_datatype values, or
after changing product_template.c in a way that affects its interface:

    python3 reference_implementation/src/generate_product_instantiations.py \
        > reference_implementation/src/product_instantiations.gen.c
"""

# (TAPP_datatype enum name, C type used for that datatype, CMake option that
# must be defined for the type to be available) - order matches the enum
# values in api/include/tapp/datatype.h (F32=0 .. BF16=5). F16/BF16 are
# guarded because not every compiler/target supports _Float16 or __bf16.
DATATYPES = [
    ("TAPP_F32", "float", None),
    ("TAPP_F64", "double", None),
    ("TAPP_C32", "scomplex", None),
    ("TAPP_C64", "dcomplex", None),
    ("TAPP_F16", "_Float16", "TAPP_REFERENCE_ENABLE_F16"),
    ("TAPP_BF16", "__bf16", "TAPP_REFERENCE_ENABLE_BF16"),
]

HEADER = """\
/*
 * GENERATED FILE - do not edit by hand.
 * Regenerate with generate_product_instantiations.py (see that file's
 * docstring). Included from product.c.
 */
"""


def guards_for(*guard_lists):
    """Dedupe and order the CMake-option guards required by a combination of
    datatypes, from the per-slot guard (or None) of each of T_A..T_D."""
    guards = []
    for g in guard_lists:
        if g and g not in guards:
            guards.append(g)
    return guards


def guard_condition(guards):
    return " && ".join(f"defined({g})" for g in guards)


def instantiation_block(t_a, t_b, t_c, t_d, guards):
    lines = []
    if guards:
        lines.append(f"#if {guard_condition(guards)}\n")
    lines.append(
        f"#define T_A {t_a}\n"
        f"#define T_B {t_b}\n"
        f"#define T_C {t_c}\n"
        f"#define T_D {t_d}\n"
        f"#include \"product_template.c\"\n"
        f"#undef T_A\n"
        f"#undef T_B\n"
        f"#undef T_C\n"
        f"#undef T_D\n"
    )
    if guards:
        lines.append("#endif\n")
    return "".join(lines)


def main():
    parts = [HEADER]

    for ea, t_a, ga in DATATYPES:
        for eb, t_b, gb in DATATYPES:
            for ec, t_c, gc in DATATYPES:
                for ed, t_d, gd in DATATYPES:
                    guards = guards_for(ga, gb, gc, gd)
                    parts.append(instantiation_block(t_a, t_b, t_c, t_d, guards))

    n = len(DATATYPES)
    parts.append(f"\n#define TAPP_PRODUCT_NUM_DATATYPES {n}\n")
    parts.append(
        "\n"
        "static const TAPP_execute_product_fn TAPP_PRODUCT_DISPATCH"
        f"[{n}][{n}][{n}][{n}] = {{\n"
    )
    for ea, t_a, ga in DATATYPES:
        parts.append(f"    [{ea}] = {{\n")
        for eb, t_b, gb in DATATYPES:
            parts.append(f"        [{eb}] = {{\n")
            for ec, t_c, gc in DATATYPES:
                parts.append(f"            [{ec}] = {{\n")
                for ed, t_d, gd in DATATYPES:
                    guards = guards_for(ga, gb, gc, gd)
                    fn = f"TAPP_EXECUTE_PRODUCT_TEMPLATE({t_a}, {t_b}, {t_c}, {t_d})"
                    entry = f"                [{ed}] = {fn},\n"
                    if guards:
                        entry = (
                            f"#if {guard_condition(guards)}\n"
                            f"{entry}"
                            "#endif\n"
                        )
                    parts.append(entry)
                parts.append("            },\n")
            parts.append("        },\n")
        parts.append("    },\n")
    parts.append("};\n")

    print("".join(parts), end="")


if __name__ == "__main__":
    main()

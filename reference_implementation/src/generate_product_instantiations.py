#!/usr/bin/env python3
"""
Generates product_instantiations.gen.c: one TAPP_EXECUTE_PRODUCT_TEMPLATE
instantiation per supported (type_A, type_B, type_C, type_D, prec)
combination, plus the TAPP_PRODUCT_DISPATCH[] table used by
TAPP_execute_product to pick the right one at runtime.

Only a small, explicit set of combinations is supported (see COMBOS below),
not the full cross product of datatypes - most (T_A, T_B, T_C, T_D, prec)
combinations don't correspond to anything a caller would request, and the
accumulator type (T_ACC) is itself part of what a combination selects, so
the full cross product would blow up combinatorially for no benefit. Add
new entries to COMBOS to support more combinations later.

Regenerate after changing COMBOS, or after changing product_template.c in a
way that affects its interface:

    python3 reference_implementation/src/generate_product_instantiations.py \
        > reference_implementation/src/product_instantiations.gen.c
"""

# C type -> (TAPP_datatype enum name, CMake option that must be defined for
# the type to be available, or None if always available). F16/BF16 are
# guarded because not every compiler/target supports _Float16 or __bf16.
DATATYPES = {
    "float": ("TAPP_F32", None),
    "double": ("TAPP_F64", None),
    "scomplex": ("TAPP_C32", None),
    "dcomplex": ("TAPP_C64", None),
    "_Float16": ("TAPP_F16", "TAPP_REFERENCE_ENABLE_F16"),
    "__bf16": ("TAPP_BF16", "TAPP_REFERENCE_ENABLE_BF16"),
}

# (T_A, T_B, T_C, T_D, T_ACC, TAPP_prectype enum name) for every supported
# combination:
#   - Uniform 32/64-bit real/complex: storage and accumulator match.
#   - Mixed data type: A, C real (float64); B, D complex (complex128).
#   - Mixed precision: 32-bit real/complex storage, 64-bit accumulator,
#     selected explicitly via TAPP_F64F64_ACCUM_F64 (no dedicated prectype
#     value exists for "32-bit storage, 64-bit accumulate" - reusing the
#     64-bit-accumulate constant is how a caller asks for it here).
COMBOS = [
    ("float", "float", "float", "float", "float", "TAPP_DEFAULT_PREC"),
    ("double", "double", "double", "double", "double", "TAPP_DEFAULT_PREC"),
    ("scomplex", "scomplex", "scomplex", "scomplex", "scomplex", "TAPP_DEFAULT_PREC"),
    ("dcomplex", "dcomplex", "dcomplex", "dcomplex", "dcomplex", "TAPP_DEFAULT_PREC"),

    ("double", "dcomplex", "double", "dcomplex", "dcomplex", "TAPP_DEFAULT_PREC"),

    ("float", "float", "float", "float", "double", "TAPP_F64F64_ACCUM_F64"),
    ("scomplex", "scomplex", "scomplex", "scomplex", "dcomplex", "TAPP_F64F64_ACCUM_F64"),
]

HEADER = """\
/*
 * GENERATED FILE - do not edit by hand.
 * Regenerate with generate_product_instantiations.py (see that file's
 * docstring). Included from product.c.
 */
"""


def guards_for(*types):
    """Dedupe and order the CMake-option guards required by a combination of
    C types."""
    guards = []
    for t in types:
        g = DATATYPES[t][1]
        if g and g not in guards:
            guards.append(g)
    return guards


def guard_condition(guards):
    return " && ".join(f"defined({g})" for g in guards)


def instantiation_block(t_a, t_b, t_c, t_d, t_acc, guards):
    lines = []
    if guards:
        lines.append(f"#if {guard_condition(guards)}\n")
    lines.append(
        f"#define T_A {t_a}\n"
        f"#define T_B {t_b}\n"
        f"#define T_C {t_c}\n"
        f"#define T_D {t_d}\n"
        f"#define T_ACC {t_acc}\n"
        f"#include \"product_template.c\"\n"
        f"#undef T_A\n"
        f"#undef T_B\n"
        f"#undef T_C\n"
        f"#undef T_D\n"
        f"#undef T_ACC\n"
    )
    if guards:
        lines.append("#endif\n")
    return "".join(lines)


def main():
    parts = [HEADER]

    for t_a, t_b, t_c, t_d, t_acc, prec in COMBOS:
        guards = guards_for(t_a, t_b, t_c, t_d, t_acc)
        parts.append(instantiation_block(t_a, t_b, t_c, t_d, t_acc, guards))

    parts.append(
        "\n"
        "static const TAPP_product_dispatch_entry TAPP_PRODUCT_DISPATCH[] = {\n"
    )
    for t_a, t_b, t_c, t_d, t_acc, prec in COMBOS:
        guards = guards_for(t_a, t_b, t_c, t_d, t_acc)
        e_a, e_b, e_c, e_d = (DATATYPES[t][0] for t in (t_a, t_b, t_c, t_d))
        fn = f"TAPP_EXECUTE_PRODUCT_TEMPLATE({t_a}, {t_b}, {t_c}, {t_d}, {t_acc})"
        entry = f"    {{ {e_a}, {e_b}, {e_c}, {e_d}, {prec}, {fn} }},\n"
        if guards:
            entry = (
                f"#if {guard_condition(guards)}\n"
                f"{entry}"
                "#endif\n"
            )
        parts.append(entry)
    parts.append("};\n")
    parts.append(
        "\n"
        "#define TAPP_PRODUCT_DISPATCH_COUNT "
        "(sizeof(TAPP_PRODUCT_DISPATCH) / sizeof(TAPP_PRODUCT_DISPATCH[0]))\n"
    )

    print("".join(parts), end="")


if __name__ == "__main__":
    main()

from .pipeline import IRPassContext


class VerifyIRPass:
    name = "verify"

    def run(self, ir_text: str, ctx: IRPassContext) -> str:
        _ = ctx
        import llvmlite.binding as llvm

        module = llvm.parse_assembly(ir_text)
        module.verify()
        return ir_text

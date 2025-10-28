import ida_hexrays
import ida_hexrays as hr
import ida_typeinf
import ida_idaapi
import ida_funcs
import ida_bytes
import ida_name
import ida_kernwin
import idaapi
import idautils
import struct
import re
from typing import Dict, List, Tuple, Optional, Set, Any
import llvmlite.binding as llvm
from llvmlite import ir
import logging

# Initialize LLVM
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class LLVMToMicrocodeConverter:
    """Converts LLVM IR to IDA Microcode"""
    
    def __init__(self):
        self.module = None
        self.functions = {}
        self.globals = {}
        self.type_cache = {}
        self.mba_cache = {}
        self.lvar_map = {}
        self.block_map = {}
        self.insn_map = {}
        self.current_func = None
        self.current_mba = None
        self.current_block = None
        self.temp_counter = 0
        self.ptrsize = 64 if ida_idaapi.get_inf_structure().is_64bit() else 32
        
    def parse_llvm_ir(self, ir_text: str) -> llvm.ModuleRef:
        """Parse LLVM IR text into module"""
        try:
            self.module = llvm.parse_assembly(ir_text)
            return self.module
        except Exception as e:
            logging.error(f"Failed to parse LLVM IR: {e}")
            raise
            
    def convert_module(self, ir_text: str):
        """Main conversion function"""
        # Parse LLVM IR
        self.parse_llvm_ir(ir_text)
        
        # Extract globals
        self._extract_globals()
        
        # Convert each function
        for func in self.module.functions:
            if not func.is_declaration:
                self._convert_function(func)
                
    def _extract_globals(self):
        """Extract global variables from LLVM module"""
        for gv in self.module.global_variables:
            self.globals[gv.name] = {
                'type': self._convert_llvm_type_to_ida(str(gv.type)),
                'initializer': gv.initializer if hasattr(gv, 'initializer') else None,
                'address': self._allocate_global_address(gv.name)
            }
            
    def _allocate_global_address(self, name: str) -> int:
        """Allocate address for global variable"""
        # Try to find existing address
        ea = ida_name.get_name_ea(ida_idaapi.BADADDR, name)
        if ea != ida_idaapi.BADADDR:
            return ea
        # Otherwise allocate new one (this is simplified)
        return 0x10000000 + len(self.globals) * 0x1000
        
    def _convert_function(self, llvm_func):
        """Convert LLVM function to IDA microcode"""
        func_name = llvm_func.name
        func_ea = ida_name.get_name_ea(ida_idaapi.BADADDR, func_name)
        
        if func_ea == ida_idaapi.BADADDR:
            # Create new function
            func_ea = self._create_function_stub(func_name)
            
        # Get or create microcode
        hf = ida_hexrays.hexrays_failure_t()
        mbr = ida_hexrays.mba_ranges_t()
        mbr.ranges.push_back(ida_funcs.get_func(func_ea))
        
        # Create microcode
        mba = ida_hexrays.gen_microcode(mbr, hf, None, ida_hexrays.DECOMP_NO_WAIT)
        if not mba:
            logging.error(f"Failed to create microcode for {func_name}")
            return
            
        self.current_func = llvm_func
        self.current_mba = mba
        self.lvar_map[func_name] = {}
        self.block_map[func_name] = {}
        
        # Convert function signature
        self._convert_function_signature(llvm_func, mba)
        
        # Convert basic blocks
        for bb_idx, bb in enumerate(llvm_func.basic_blocks):
            self._convert_basic_block(bb, bb_idx)
            
        # Analyze and optimize
        mba.analyze()
        
        # Save to cache
        self.mba_cache[func_name] = mba
        self.functions[func_name] = func_ea
        
    def _create_function_stub(self, name: str) -> int:
        """Create a stub function in IDA"""
        # This is simplified - in practice you'd need to allocate proper memory
        stub_ea = 0x1000 + len(self.functions) * 0x100
        ida_funcs.add_func(stub_ea, stub_ea + 0x10)
        ida_name.set_name(stub_ea, name, ida_name.SN_NOWARN)
        return stub_ea
        
    def _convert_function_signature(self, llvm_func, mba):
        """Convert LLVM function signature to IDA microcode"""
        # Parse function type
        func_type = llvm_func.type
        
        # Convert return type
        ret_tif = self._convert_llvm_type_to_ida(str(llvm_func.return_type))
        
        # Convert arguments
        for idx, arg in enumerate(llvm_func.args):
            arg_tif = self._convert_llvm_type_to_ida(str(arg.type))
            
            # Create lvar for argument
            lvar = self._create_lvar(mba, arg.name, arg_tif, is_arg=True)
            self.lvar_map[llvm_func.name][arg.name] = lvar
            
    def _convert_basic_block(self, bb, bb_idx: int):
        """Convert LLVM basic block to microcode block"""
        # Get or create microcode block
        mblock = self._get_or_create_mblock(bb_idx)
        self.current_block = mblock
        self.block_map[self.current_func.name][bb.name] = bb_idx
        
        # Convert each instruction
        for insn in bb.instructions:
            self._convert_instruction(insn, mblock)
            
    def _get_or_create_mblock(self, idx: int):
        """Get or create microcode block"""
        if idx < self.current_mba.qty:
            return self.current_mba.get_mblock(idx)
        else:
            # Create new block
            return self.current_mba.insert_block(idx)
            
    def _convert_instruction(self, llvm_insn, mblock):
        """Convert LLVM instruction to microcode instruction"""
        opcode = llvm_insn.opcode
        
        # Dispatch based on instruction type
        if opcode == 'alloca':
            self._convert_alloca(llvm_insn, mblock)
        elif opcode == 'load':
            self._convert_load(llvm_insn, mblock)
        elif opcode == 'store':
            self._convert_store(llvm_insn, mblock)
        elif opcode in ['add', 'sub', 'mul', 'udiv', 'sdiv', 'urem', 'srem']:
            self._convert_binary_op(llvm_insn, mblock)
        elif opcode in ['and', 'or', 'xor', 'shl', 'lshr', 'ashr']:
            self._convert_bitwise_op(llvm_insn, mblock)
        elif opcode in ['icmp', 'fcmp']:
            self._convert_compare(llvm_insn, mblock)
        elif opcode == 'br':
            self._convert_branch(llvm_insn, mblock)
        elif opcode == 'call':
            self._convert_call(llvm_insn, mblock)
        elif opcode == 'ret':
            self._convert_return(llvm_insn, mblock)
        elif opcode == 'phi':
            self._convert_phi(llvm_insn, mblock)
        elif opcode in ['zext', 'sext', 'trunc']:
            self._convert_cast(llvm_insn, mblock)
        elif opcode in ['bitcast', 'inttoptr', 'ptrtoint']:
            self._convert_ptr_cast(llvm_insn, mblock)
        elif opcode == 'getelementptr':
            self._convert_gep(llvm_insn, mblock)
        elif opcode == 'select':
            self._convert_select(llvm_insn, mblock)
        else:
            logging.warning(f"Unhandled instruction: {opcode}")
            
    def _convert_alloca(self, llvm_insn, mblock):
        """Convert alloca instruction"""
        # Extract type and create stack variable
        allocated_type = self._extract_alloca_type(llvm_insn)
        tif = self._convert_llvm_type_to_ida(allocated_type)
        
        # Create lvar
        lvar = self._create_lvar(self.current_mba, llvm_insn.name, tif, is_stack=True)
        self.lvar_map[self.current_func.name][llvm_insn.name] = lvar
        
        # No microcode needed for alloca itself
        
    def _convert_load(self, llvm_insn, mblock):
        """Convert load instruction"""
        # Get source operand
        src_op = self._convert_operand(llvm_insn.operands[0])
        
        # Create destination
        dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
        dst_op = self._create_lvar_mop(dst_lvar)
        
        # Create ldx instruction
        size = self._get_type_size(llvm_insn.type)
        minsn = self._create_minsn(hr.m_ldx, size)
        minsn.l = src_op  # address
        minsn.r = self._create_num_mop(0)  # offset
        minsn.d = dst_op
        
        self._append_minsn(mblock, minsn)
        
    def _convert_store(self, llvm_insn, mblock):
        """Convert store instruction"""
        # Get value and destination
        value_op = self._convert_operand(llvm_insn.operands[0])
        dst_op = self._convert_operand(llvm_insn.operands[1])
        
        # Create stx instruction
        size = self._get_operand_size(value_op)
        minsn = self._create_minsn(hr.m_stx, size)
        minsn.l = value_op
        minsn.r = self._create_num_mop(0)  # offset
        minsn.d = dst_op
        
        self._append_minsn(mblock, minsn)
        
    def _convert_binary_op(self, llvm_insn, mblock):
        """Convert binary arithmetic operation"""
        op_map = {
            'add': hr.m_add,
            'sub': hr.m_sub,
            'mul': hr.m_mul,
            'udiv': hr.m_udiv,
            'sdiv': hr.m_sdiv,
            'urem': hr.m_umod,
            'srem': hr.m_smod,
        }
        
        # Get operands
        lhs = self._convert_operand(llvm_insn.operands[0])
        rhs = self._convert_operand(llvm_insn.operands[1])
        
        # Create result
        dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
        dst = self._create_lvar_mop(dst_lvar)
        
        # Create instruction
        mcode = op_map.get(llvm_insn.opcode)
        size = self._get_type_size(llvm_insn.type)
        minsn = self._create_minsn(mcode, size)
        minsn.l = lhs
        minsn.r = rhs
        minsn.d = dst
        
        self._append_minsn(mblock, minsn)
        
    def _convert_compare(self, llvm_insn, mblock):
        """Convert comparison instruction"""
        predicate = llvm_insn.predicate
        
        # Map LLVM predicates to IDA microcode
        pred_map = {
            'eq': hr.m_setz,
            'ne': hr.m_setnz,
            'ugt': hr.m_seta,
            'uge': hr.m_setae,
            'ult': hr.m_setb,
            'ule': hr.m_setbe,
            'sgt': hr.m_setg,
            'sge': hr.m_setge,
            'slt': hr.m_setl,
            'sle': hr.m_setle,
        }
        
        # Get operands
        lhs = self._convert_operand(llvm_insn.operands[0])
        rhs = self._convert_operand(llvm_insn.operands[1])
        
        # Create result
        dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
        dst = self._create_lvar_mop(dst_lvar)
        
        # Create set instruction
        mcode = pred_map.get(predicate, hr.m_setz)
        size = self._get_operand_size(lhs)
        minsn = self._create_minsn(mcode, size)
        minsn.l = lhs
        minsn.r = rhs
        minsn.d = dst
        
        self._append_minsn(mblock, minsn)
        
    def _convert_branch(self, llvm_insn, mblock):
        """Convert branch instruction"""
        if len(llvm_insn.operands) == 1:
            # Unconditional branch
            target_bb = llvm_insn.operands[0].name
            target_idx = self._get_block_index(target_bb)
            
            minsn = self._create_minsn(hr.m_goto, 0)
            minsn.l = self._create_block_mop(target_idx)
            
            self._append_minsn(mblock, minsn)
        else:
            # Conditional branch
            cond = self._convert_operand(llvm_insn.operands[0])
            true_bb = llvm_insn.operands[1].name
            false_bb = llvm_insn.operands[2].name
            
            true_idx = self._get_block_index(true_bb)
            false_idx = self._get_block_index(false_bb)
            
            # Create conditional jump
            minsn = self._create_minsn(hr.m_jcnd, 1)
            minsn.l = cond
            minsn.d = self._create_block_mop(true_idx)
            
            self._append_minsn(mblock, minsn)
            
            # Add goto for false branch
            goto_insn = self._create_minsn(hr.m_goto, 0)
            goto_insn.l = self._create_block_mop(false_idx)
            self._append_minsn(mblock, goto_insn)
            
    def _convert_call(self, llvm_insn, mblock):
        """Convert call instruction"""
        # Get function
        func_op = llvm_insn.operands[-1]  # Last operand is function
        
        # Get arguments
        args = []
        for i in range(len(llvm_insn.operands) - 1):
            arg_op = self._convert_operand(llvm_insn.operands[i])
            args.append(arg_op)
            
        # Create call info
        call_info = self._create_call_info(args)
        
        # Create result if not void
        dst_op = None
        if llvm_insn.type != 'void':
            dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
            dst_op = self._create_lvar_mop(dst_lvar)
            call_info.return_regs.push_back(dst_op)
            
        # Create call instruction
        minsn = self._create_minsn(hr.m_call, 0)
        minsn.l = self._convert_operand(func_op)
        minsn.d = self._create_call_mop(call_info)
        
        self._append_minsn(mblock, minsn)
        
    def _convert_return(self, llvm_insn, mblock):
        """Convert return instruction"""
        if len(llvm_insn.operands) > 0:
            # Return with value
            ret_val = self._convert_operand(llvm_insn.operands[0])
            
            # Store to result variable
            result_lvar = self._get_result_lvar()
            if result_lvar:
                store_insn = self._create_minsn(hr.m_stx, self._get_operand_size(ret_val))
                store_insn.l = ret_val
                store_insn.d = self._create_lvar_mop(result_lvar)
                self._append_minsn(mblock, store_insn)
                
        # Create ret instruction
        ret_insn = self._create_minsn(hr.m_ret, 0)
        self._append_minsn(mblock, ret_insn)
        
    def _convert_cast(self, llvm_insn, mblock):
        """Convert cast instruction (zext, sext, trunc)"""
        src = self._convert_operand(llvm_insn.operands[0])
        dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
        dst = self._create_lvar_mop(dst_lvar)
        
        # Determine cast type
        if llvm_insn.opcode == 'zext':
            mcode = hr.m_xdu
        elif llvm_insn.opcode == 'sext':
            mcode = hr.m_xds
        elif llvm_insn.opcode == 'trunc':
            mcode = hr.m_low
        else:
            mcode = hr.m_mov
            
        size = self._get_type_size(llvm_insn.type)
        minsn = self._create_minsn(mcode, size)
        minsn.l = src
        minsn.d = dst
        
        self._append_minsn(mblock, minsn)
        
    def _convert_gep(self, llvm_insn, mblock):
        """Convert getelementptr instruction"""
        # Get base pointer
        base = self._convert_operand(llvm_insn.operands[0])
        
        # Calculate offset
        offset = 0
        current_type = str(llvm_insn.operands[0].type)
        
        # Process indices
        for idx_op in llvm_insn.operands[1:]:
            if self._is_constant(idx_op):
                idx_val = self._get_constant_value(idx_op)
                # Calculate offset based on type
                # This is simplified - real implementation needs proper type handling
                offset += idx_val * 8  # Assume 8 bytes for now
                
        # Create result
        dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
        dst = self._create_lvar_mop(dst_lvar)
        
        if offset == 0:
            # Simple move
            minsn = self._create_minsn(hr.m_mov, self.ptrsize // 8)
            minsn.l = base
            minsn.d = dst
        else:
            # Add offset
            minsn = self._create_minsn(hr.m_add, self.ptrsize // 8)
            minsn.l = base
            minsn.r = self._create_num_mop(offset)
            minsn.d = dst
            
        self._append_minsn(mblock, minsn)
        
    def _convert_phi(self, llvm_insn, mblock):
        """Convert PHI instruction"""
        # PHI nodes require special handling
        # For now, create a temporary variable
        dst_lvar = self._get_or_create_temp_lvar(llvm_insn)
        
        # PHI resolution happens at block boundaries
        # This is a simplified approach
        logging.warning("PHI node conversion is simplified")
        
    def _convert_operand(self, operand) -> 'mop_t':
        """Convert LLVM operand to microcode operand"""
        op_str = str(operand)
        
        # Check if it's a constant
        if self._is_constant(operand):
            value = self._get_constant_value(operand)
            return self._create_num_mop(value)
            
        # Check if it's a local variable reference
        if op_str.startswith('%'):
            var_name = op_str[1:]
            if var_name in self.lvar_map.get(self.current_func.name, {}):
                lvar = self.lvar_map[self.current_func.name][var_name]
                return self._create_lvar_mop(lvar)
                
        # Check if it's a global reference
        if op_str.startswith('@'):
            global_name = op_str[1:]
            return self._create_global_mop(global_name)
            
        # Default: create temporary
        return self._create_num_mop(0)
        
    def _create_minsn(self, opcode: int, size: int) -> 'minsn_t':
        """Create microcode instruction"""
        minsn = hr.minsn_t(0)  # ea = 0 for now
        minsn.opcode = opcode
        minsn.l.size = size
        minsn.r.size = size
        minsn.d.size = size
        return minsn
        
    def _create_lvar_mop(self, lvar) -> 'mop_t':
        """Create local variable operand"""
        mop = hr.mop_t()
        mop.t = hr.mop_l
        mop.size = lvar.width // 8
        mop.l = lvar
        return mop
        
    def _create_num_mop(self, value: int) -> 'mop_t':
        """Create number operand"""
        mop = hr.mop_t()
        mop.t = hr.mop_n
        mop.nnn = hr.mnumber_t(value)
        mop.size = 8  # Default size
        return mop
        
    def _create_global_mop(self, name: str) -> 'mop_t':
        """Create global variable operand"""
        mop = hr.mop_t()
        mop.t = hr.mop_v
        if name in self.globals:
            mop.g = self.globals[name]['address']
        else:
            mop.g = 0  # Unknown global
        mop.size = 8
        return mop
        
    def _create_block_mop(self, block_idx: int) -> 'mop_t':
        """Create block reference operand"""
        mop = hr.mop_t()
        mop.t = hr.mop_b
        mop.b = block_idx
        return mop
        
    def _create_call_mop(self, call_info) -> 'mop_t':
        """Create call info operand"""
        mop = hr.mop_t()
        mop.t = hr.mop_f
        mop.f = call_info
        return mop
        
    def _create_lvar(self, mba, name: str, tif, is_arg=False, is_stack=False):
        """Create local variable"""
        lvar = hr.lvar_t()
        lvar.name = name
        lvar.type = tif
        
        if is_arg:
            lvar.set_arg_var()
        elif is_stack:
            lvar.set_stk_var()
            
        # Add to mba
        mba.vars.push_back(lvar)
        return lvar
        
    def _get_or_create_temp_lvar(self, llvm_insn):
        """Get or create temporary variable for instruction result"""
        name = llvm_insn.name if hasattr(llvm_insn, 'name') and llvm_insn.name else f"tmp_{self.temp_counter}"
        self.temp_counter += 1
        
        if name in self.lvar_map.get(self.current_func.name, {}):
            return self.lvar_map[self.current_func.name][name]
            
        # Create new temp var
        tif = self._convert_llvm_type_to_ida(str(llvm_insn.type))
        lvar = self._create_lvar(self.current_mba, name, tif)
        self.lvar_map[self.current_func.name][name] = lvar
        return lvar
        
    def _get_result_lvar(self):
        """Get function result variable"""
        for lvar in self.current_mba.vars:
            if lvar.is_result_var:
                return lvar
        return None
        
    def _append_minsn(self, mblock, minsn):
        """Append instruction to block"""
        if mblock.tail:
            mblock.tail.next = minsn
            minsn.prev = mblock.tail
            mblock.tail = minsn
        else:
            mblock.head = mblock.tail = minsn
            
    def _convert_llvm_type_to_ida(self, llvm_type_str: str) -> 'tinfo_t':
        """Convert LLVM type string to IDA type"""
        tif = ida_typeinf.tinfo_t()
        
        # Remove pointer markers
        ptr_count = llvm_type_str.count('*')
        base_type = llvm_type_str.replace('*', '').strip()
        
        # Convert base type
        if base_type == 'void':
            tif.create_simple_type(ida_typeinf.BTF_VOID)
        elif base_type == 'i1':
            tif.create_simple_type(ida_typeinf.BTF_BOOL)
        elif base_type == 'i8':
            tif.create_simple_type(ida_typeinf.BTF_CHAR)
        elif base_type == 'i16':
            tif.create_simple_type(ida_typeinf.BTF_INT16)
        elif base_type == 'i32':
            tif.create_simple_type(ida_typeinf.BTF_INT32)
        elif base_type == 'i64':
            tif.create_simple_type(ida_typeinf.BTF_INT64)
        elif base_type == 'float':
            tif.create_simple_type(ida_typeinf.BTF_FLOAT)
        elif base_type == 'double':
            tif.create_simple_type(ida_typeinf.BTF_DOUBLE)
        else:
            # Default to int
            tif.create_simple_type(ida_typeinf.BTF_INT)
            
        # Apply pointer indirection
        for _ in range(ptr_count):
            ptr_tif = ida_typeinf.tinfo_t()
            ptr_tif.create_ptr(tif)
            tif = ptr_tif
            
        return tif
        
    def _get_type_size(self, llvm_type) -> int:
        """Get size of LLVM type in bytes"""
        type_str = str(llvm_type)
        
        if 'i1' in type_str:
            return 1
        elif 'i8' in type_str:
            return 1
        elif 'i16' in type_str:
            return 2
        elif 'i32' in type_str:
            return 4
        elif 'i64' in type_str:
            return 8
        elif 'float' in type_str:
            return 4
        elif 'double' in type_str:
            return 8
        elif '*' in type_str:
            return self.ptrsize // 8
        else:
            return 8  # Default
            
    def _get_operand_size(self, mop) -> int:
        """Get size of microcode operand"""
        return mop.size if hasattr(mop, 'size') else 8
        
    def _is_constant(self, operand) -> bool:
        """Check if operand is a constant"""
        op_str = str(operand)
        return op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit())
        
    def _get_constant_value(self, operand) -> int:
        """Extract constant value from operand"""
        op_str = str(operand)
        if op_str.isdigit() or (op_str.startswith('-') and op_str[1:].isdigit()):
            return int(op_str)
        return 0
        
    def _get_block_index(self, bb_name: str) -> int:
        """Get block index from name"""
        if bb_name in self.block_map.get(self.current_func.name, {}):
            return self.block_map[self.current_func.name][bb_name]
        # Allocate new block
        new_idx = self.current_mba.qty
        self.block_map[self.current_func.name][bb_name] = new_idx
        return new_idx
        
    def _extract_alloca_type(self, llvm_insn) -> str:
        """Extract allocated type from alloca instruction"""
        # Parse instruction string to get type
        insn_str = str(llvm_insn)
        match = re.search(r'alloca\s+([^,]+)', insn_str)
        if match:
            return match.group(1).strip()
        return 'i8'
        
    def _create_call_info(self, args) -> 'mcallinfo_t':
        """Create call info structure"""
        ci = hr.mcallinfo_t()
        for arg in args:
            ci.args.push_back(arg)
        return ci

    def apply_to_database(self):
        """Apply converted microcode to IDA database"""
        for func_name, mba in self.mba_cache.items():
            func_ea = self.functions[func_name]
            
            # Force recompilation with new microcode
            ida_hexrays.mark_cfunc_dirty(func_ea)
            
            # Decompile to apply changes
            cfunc = ida_hexrays.decompile(func_ea)
            if cfunc:
                logging.info(f"Successfully applied microcode for {func_name}")
            else:
                logging.error(f"Failed to decompile {func_name}")

def main():
    """Main function to run the converter"""
    # Example usage
    converter = LLVMToMicrocodeConverter()
    
    # Read LLVM IR from file
    with open("input.ll", "r") as f:
        llvm_ir = f.read()
        
    # Convert
    converter.convert_module(llvm_ir)
    
    # Apply to database
    converter.apply_to_database()
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
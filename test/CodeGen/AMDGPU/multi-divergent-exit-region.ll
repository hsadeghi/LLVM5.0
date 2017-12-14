; RUN: opt -mtriple=amdgcn-- -S -amdgpu-unify-divergent-exit-nodes -verify -structurizecfg -verify -si-annotate-control-flow %s | FileCheck -check-prefix=IR %s
; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Add an extra verifier runs. There were some cases where invalid IR
; was produced but happened to be fixed by the later passes.

; Make sure divergent control flow with multiple exits from a region
; is properly handled. UnifyFunctionExitNodes should be run before
; StructurizeCFG.

; IR-LABEL: @multi_divergent_region_exit_ret_ret(
; IR: %Pivot = icmp sge i32 %tmp16, 2
; IR-NEXT: %0 = call { i1, i64 } @llvm.amdgcn.if(i1 %Pivot)
; IR: %1 = extractvalue { i1, i64 } %0, 0
; IR: %2 = extractvalue { i1, i64 } %0, 1
; IR: br i1 %1, label %LeafBlock1, label %Flow

; IR: Flow:
; IR: %3 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %4 = phi i1 [ %SwitchLeaf2, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = call { i1, i64 } @llvm.amdgcn.else(i64 %2)
; IR: %6 = extractvalue { i1, i64 } %5, 0
; IR: %7 = extractvalue { i1, i64 } %5, 1
; IR: br i1 %6, label %LeafBlock, label %Flow1

; IR: LeafBlock:
; IR: br label %Flow1

; IR: LeafBlock1:
; IR: br label %Flow{{$}}

; IR:  Flow2:
; IR: %8 = phi i1 [ false, %exit1 ], [ %12, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %16)
; IR: [[IF:%[0-9]+]] = call { i1, i64 } @llvm.amdgcn.if(i1 %8)
; IR: %10 = extractvalue { i1, i64 } [[IF]], 0
; IR: %11 = extractvalue { i1, i64 } [[IF]], 1
; IR: br i1 %10, label %exit0, label %UnifiedReturnBlock

; IR: exit0:
; IR: store volatile i32 9, i32 addrspace(1)* undef
; IR: br label %UnifiedReturnBlock

; IR: Flow1:
; IR: %12 = phi i1 [ %SwitchLeaf, %LeafBlock ], [ %3, %Flow ]
; IR: %13 = phi i1 [ %SwitchLeaf, %LeafBlock ], [ %4, %Flow ]
; IR: call void @llvm.amdgcn.end.cf(i64 %7)
; IR: %14 = call { i1, i64 } @llvm.amdgcn.if(i1 %13)
; IR: %15 = extractvalue { i1, i64 } %14, 0
; IR: %16 = extractvalue { i1, i64 } %14, 1
; IR: br i1 %15, label %exit1, label %Flow2

; IR: exit1:
; IR: store volatile i32 17, i32 addrspace(3)* undef
; IR:  br label %Flow2

; IR: UnifiedReturnBlock:
; IR: call void @llvm.amdgcn.end.cf(i64 %11)
; IR: ret void


; GCN-LABEL: {{^}}multi_divergent_region_exit_ret_ret:
; GCN: v_cmp_lt_i32_e32 vcc, 1
; GCN: s_and_saveexec_b64
; GCN: s_xor_b64


; GCN: ; %LeafBlock
; GCN: v_cmp_ne_u32_e32 vcc, 1, [[REG:v[0-9]+]]
; GCN: v_cndmask_b32_e64 v{{[0-9]+}}, 0, -1, vcc

; GCN: ; %Flow1
; GCN-NEXT: s_or_b64 exec, exec
; GCN: v_cmp_ne_u32_e32 vcc, 0

; GCN: ; %exit1
; GCN: ds_write_b32

; GCN: %Flow2
; GCN-NEXT: s_or_b64 exec, exec
; GCN: v_cmp_ne_u32_e32 vcc, 0
; GCN-NEXT: s_and_saveexec_b64
; GCN-NEXT: s_xor_b64

; GCN: ; %exit0
; GCN: buffer_store_dword

; GCN: ; %UnifiedReturnBlock
; GCN-NEXT: s_or_b64 exec, exec
; GCN-NEXT: s_endpgm
define amdgpu_kernel void @multi_divergent_region_exit_ret_ret(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void
}

; IR-LABEL: @multi_divergent_region_exit_unreachable_unreachable(
; IR: %Pivot = icmp sge i32 %tmp16, 2
; IR-NEXT: %0 = call { i1, i64 } @llvm.amdgcn.if(i1 %Pivot)

; IR: %5 = call { i1, i64 } @llvm.amdgcn.else(i64 %2)

; IR: %8 = phi i1 [ false, %exit1 ], [ %12, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %16)
; IR: %9 = call { i1, i64 } @llvm.amdgcn.if(i1 %8)
; IR: br i1 %10, label %exit0, label %UnifiedUnreachableBlock


; IR: UnifiedUnreachableBlock:
; IR-NEXT: unreachable


; FIXME: Probably should insert an s_endpgm anyway.
; GCN-LABEL: {{^}}multi_divergent_region_exit_unreachable_unreachable:
; GCN: ; %UnifiedUnreachableBlock
; GCN-NEXT: .Lfunc_end
define amdgpu_kernel void @multi_divergent_region_exit_unreachable_unreachable(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  unreachable
}

; IR-LABEL: @multi_exit_region_divergent_ret_uniform_ret(
; IR: %divergent.cond0 = icmp sge i32 %tmp16, 2
; IR: llvm.amdgcn.if
; IR: br i1

; IR: {{^}}Flow:
; IR: %3 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %4 = phi i1 [ %uniform.cond0, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = call { i1, i64 } @llvm.amdgcn.else(i64 %2)
; IR: br i1 %6, label %LeafBlock, label %Flow1

; IR: {{^}}LeafBlock:
; IR: %divergent.cond1 = icmp ne i32 %tmp16, 1
; IR: br label %Flow1

; IR: LeafBlock1:
; IR: %uniform.cond0 = icmp ne i32 %arg3, 2
; IR: br label %Flow

; IR: Flow2:
; IR: %8 = phi i1 [ false, %exit1 ], [ %12, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %16)
; IR: %9 = call { i1, i64 } @llvm.amdgcn.if(i1 %8)
; IR: br i1 %10, label %exit0, label %UnifiedReturnBlock

; IR: exit0:
; IR: store volatile i32 9, i32 addrspace(1)* undef
; IR: br label %UnifiedReturnBlock

; IR: {{^}}Flow1:
; IR: %12 = phi i1 [ %divergent.cond1, %LeafBlock ], [ %3, %Flow ]
; IR: %13 = phi i1 [ %divergent.cond1, %LeafBlock ], [ %4, %Flow ]
; IR: call void @llvm.amdgcn.end.cf(i64 %7)
; IR: %14 = call { i1, i64 } @llvm.amdgcn.if(i1 %13)
; IR: %15 = extractvalue { i1, i64 } %14, 0
; IR: %16 = extractvalue { i1, i64 } %14, 1
; IR: br i1 %15, label %exit1, label %Flow2

; IR: exit1:
; IR: store volatile i32 17, i32 addrspace(3)* undef
; IR: br label %Flow2

; IR: UnifiedReturnBlock:
; IR: call void @llvm.amdgcn.end.cf(i64 %11)
; IR: ret void
define amdgpu_kernel void @multi_exit_region_divergent_ret_uniform_ret(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2, i32 %arg3) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %divergent.cond0 = icmp slt i32 %tmp16, 2
  br i1 %divergent.cond0, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %divergent.cond1 = icmp eq i32 %tmp16, 1
  br i1 %divergent.cond1, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %uniform.cond0 = icmp eq i32 %arg3, 2
  br i1 %uniform.cond0, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void
}

; IR-LABEL: @multi_exit_region_uniform_ret_divergent_ret(
; IR: %Pivot = icmp sge i32 %tmp16, 2
; IR-NEXT: %0 = call { i1, i64 } @llvm.amdgcn.if(i1 %Pivot)
; IR: br i1 %1, label %LeafBlock1, label %Flow

; IR: Flow:
; IR: %3 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %4 = phi i1 [ %SwitchLeaf2, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = call { i1, i64 } @llvm.amdgcn.else(i64 %2)

; IR: %8 = phi i1 [ false, %exit1 ], [ %12, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %16)
; IR: %9 = call { i1, i64 } @llvm.amdgcn.if(i1 %8)

define amdgpu_kernel void @multi_exit_region_uniform_ret_divergent_ret(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2, i32 %arg3) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %arg3, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void
}

; IR-LABEL: @multi_divergent_region_exit_ret_ret_return_value(
; IR: Flow2:
; IR: %8 = phi float [ 2.000000e+00, %exit1 ], [ undef, %Flow1 ]
; IR: %9 = phi i1 [ false, %exit1 ], [ %13, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %17)

; IR: UnifiedReturnBlock:
; IR: %UnifiedRetVal = phi float [ %8, %Flow2 ], [ 1.000000e+00, %exit0 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %12)
; IR: ret float %UnifiedRetVal
define amdgpu_ps float @multi_divergent_region_exit_ret_ret_return_value(i32 %vgpr) #0 {
entry:
  %Pivot = icmp slt i32 %vgpr, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %vgpr, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %vgpr, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 9, i32 addrspace(1)* undef
  ret float 1.0

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 17, i32 addrspace(3)* undef
  ret float 2.0
}

; IR-LABEL: @uniform_branch_to_multi_divergent_region_exit_ret_ret_return_value(

; GCN-LABEL: {{^}}uniform_branch_to_multi_divergent_region_exit_ret_ret_return_value:
; GCN: s_cmp_gt_i32 s0, 1
; GCN: s_cbranch_scc0 [[FLOW:BB[0-9]+_[0-9]+]]

; GCN: v_cmp_ne_u32_e32 vcc, 7, v0

; GCN: {{^}}[[FLOW]]:
; GCN: s_cbranch_vccnz [[FLOW1:BB[0-9]+]]

; GCN: v_mov_b32_e32 v0, 2.0
; GCN: s_or_b64 exec, exec
; GCN: s_and_b64 exec, exec
; GCN: v_mov_b32_e32 v0, 1.0

; GCN: {{^BB[0-9]+_[0-9]+}}: ; %UnifiedReturnBlock
; GCN-NEXT: s_or_b64 exec, exec
; GCN-NEXT: ; return

define amdgpu_ps float @uniform_branch_to_multi_divergent_region_exit_ret_ret_return_value(i32 inreg %sgpr, i32 %vgpr) #0 {
entry:
  %uniform.cond = icmp slt i32 %sgpr, 2
  br i1 %uniform.cond, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %divergent.cond0 = icmp eq i32 %vgpr, 3
  br i1 %divergent.cond0, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %divergent.cond1 = icmp eq i32 %vgpr, 7
  br i1 %divergent.cond1, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 9, i32 addrspace(1)* undef
  ret float 1.0

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store i32 17, i32 addrspace(3)* undef
  ret float 2.0
}

; IR-LABEL: @multi_divergent_region_exit_ret_unreachable(
; IR: %Pivot = icmp sge i32 %tmp16, 2
; IR-NEXT: %0 = call { i1, i64 } @llvm.amdgcn.if(i1 %Pivot)

; IR: Flow:
; IR: %3 = phi i1 [ true, %LeafBlock1 ], [ false, %entry ]
; IR: %4 = phi i1 [ %SwitchLeaf2, %LeafBlock1 ], [ false, %entry ]
; IR: %5 = call { i1, i64 } @llvm.amdgcn.else(i64 %2)

; IR: Flow2:
; IR: %8 = phi i1 [ false, %exit1 ], [ %12, %Flow1 ]
; IR: call void @llvm.amdgcn.end.cf(i64 %16)
; IR: %9 = call { i1, i64 } @llvm.amdgcn.if(i1 %8)
; IR: br i1 %10, label %exit0, label %UnifiedReturnBlock

; IR: exit0:
; IR-NEXT: store volatile i32 17, i32 addrspace(3)* undef
; IR-NEXT: br label %UnifiedReturnBlock

; IR: Flow1:
; IR: %12 = phi i1 [ %SwitchLeaf, %LeafBlock ], [ %3, %Flow ]
; IR: %13 = phi i1 [ %SwitchLeaf, %LeafBlock ], [ %4, %Flow ]
; IR: call void @llvm.amdgcn.end.cf(i64 %7)
; IR: %14 = call { i1, i64 } @llvm.amdgcn.if(i1 %13)
; IR: %15 = extractvalue { i1, i64 } %14, 0
; IR: %16 = extractvalue { i1, i64 } %14, 1
; IR: br i1 %15, label %exit1, label %Flow2

; IR: exit1:
; IR-NEXT: store volatile i32 9, i32 addrspace(1)* undef
; IR-NEXT: call void @llvm.amdgcn.unreachable()
; IR-NEXT: br label %Flow2

; IR: UnifiedReturnBlock:
; IR-NEXT: call void @llvm.amdgcn.end.cf(i64 %11)
; IR-NEXT: ret void
define amdgpu_kernel void @multi_divergent_region_exit_ret_unreachable(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable
}

; The non-uniformity of the branch to the exiting blocks requires
; looking at transitive predecessors.

; IR-LABEL: @indirect_multi_divergent_region_exit_ret_unreachable(

; IR: exit0:                                            ; preds = %Flow2
; IR-NEXT: store volatile i32 17, i32 addrspace(3)* undef
; IR-NEXT: br label %UnifiedReturnBlock


; IR: indirect.exit1:
; IR: %load = load volatile i32, i32 addrspace(1)* undef
; IR: store volatile i32 %load, i32 addrspace(1)* undef
; IR: store volatile i32 9, i32 addrspace(1)* undef
; IR: call void @llvm.amdgcn.unreachable()
; IR-NEXT: br label %Flow2

; IR: UnifiedReturnBlock:                               ; preds = %exit0, %Flow2
; IR-NEXT: call void @llvm.amdgcn.end.cf(i64 %11)
; IR-NEXT: ret void
define amdgpu_kernel void @indirect_multi_divergent_region_exit_ret_unreachable(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  %Pivot = icmp slt i32 %tmp16, 2
  br i1 %Pivot, label %LeafBlock, label %LeafBlock1

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %indirect.exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %indirect.exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void

indirect.exit1:
  %load = load volatile i32, i32 addrspace(1)* undef
  store volatile i32 %load, i32 addrspace(1)* undef
  br label %exit1

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable
}

; IR-LABEL: @multi_divergent_region_exit_ret_switch(
define amdgpu_kernel void @multi_divergent_region_exit_ret_switch(i32 addrspace(1)* nocapture %arg0, i32 addrspace(1)* nocapture %arg1, i32 addrspace(1)* nocapture %arg2) #0 {
entry:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp1 = add i32 0, %tmp
  %tmp2 = zext i32 %tmp1 to i64
  %tmp3 = add i64 0, %tmp2
  %tmp4 = shl i64 %tmp3, 32
  %tmp5 = ashr exact i64 %tmp4, 32
  %tmp6 = getelementptr inbounds i32, i32 addrspace(1)* %arg0, i64 %tmp5
  %tmp7 = load i32, i32 addrspace(1)* %tmp6, align 4
  %tmp8 = sext i32 %tmp7 to i64
  %tmp9 = getelementptr inbounds i32, i32 addrspace(1)* %arg1, i64 %tmp8
  %tmp10 = load i32, i32 addrspace(1)* %tmp9, align 4
  %tmp13 = zext i32 %tmp10 to i64
  %tmp14 = getelementptr inbounds i32, i32 addrspace(1)* %arg2, i64 %tmp13
  %tmp16 = load i32, i32 addrspace(1)* %tmp14, align 16
  switch i32 %tmp16, label %exit1
    [ i32 1, label %LeafBlock
      i32 2, label %LeafBlock1
      i32 3, label %exit0 ]

LeafBlock:                                        ; preds = %entry
  %SwitchLeaf = icmp eq i32 %tmp16, 1
  br i1 %SwitchLeaf, label %exit0, label %exit1

LeafBlock1:                                       ; preds = %entry
  %SwitchLeaf2 = icmp eq i32 %tmp16, 2
  br i1 %SwitchLeaf2, label %exit0, label %exit1

exit0:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 17, i32 addrspace(3)* undef
  ret void

exit1:                                     ; preds = %LeafBlock, %LeafBlock1
  store volatile i32 9, i32 addrspace(1)* undef
  unreachable
}

; IR-LABEL: @divergent_multi_ret_nest_in_uniform_triangle(
define amdgpu_kernel void @divergent_multi_ret_nest_in_uniform_triangle(i32 %arg0) #0 {
entry:
  %uniform.cond0 = icmp eq i32 %arg0, 4
  br i1 %uniform.cond0, label %divergent.multi.exit.region, label %uniform.ret

divergent.multi.exit.region:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %divergent.cond0 = icmp eq i32 %id.x, 0
  br i1 %divergent.cond0, label %divergent.ret0, label %divergent.ret1

divergent.ret0:
  store volatile i32 11, i32 addrspace(3)* undef
  ret void

divergent.ret1:
  store volatile i32 42, i32 addrspace(3)* undef
  ret void

uniform.ret:
  store volatile i32 9, i32 addrspace(1)* undef
  ret void
}

; IR-LABEL: @divergent_complex_multi_ret_nest_in_uniform_triangle(
define amdgpu_kernel void @divergent_complex_multi_ret_nest_in_uniform_triangle(i32 %arg0) #0 {
entry:
  %uniform.cond0 = icmp eq i32 %arg0, 4
  br i1 %uniform.cond0, label %divergent.multi.exit.region, label %uniform.ret

divergent.multi.exit.region:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %divergent.cond0 = icmp eq i32 %id.x, 0
  br i1 %divergent.cond0, label %divergent.if, label %divergent.ret1

divergent.if:
  %vgpr0 = load volatile float, float addrspace(1)* undef
  %divergent.cond1 = fcmp ogt float %vgpr0, 1.0
  br i1 %divergent.cond1, label %divergent.then, label %divergent.endif

divergent.then:
  %vgpr1 = load volatile float, float addrspace(1)* undef
  %divergent.cond2 = fcmp olt float %vgpr1, 4.0
  store volatile i32 33, i32 addrspace(1)* undef
  br i1 %divergent.cond2, label %divergent.ret0, label %divergent.endif

divergent.endif:
  store volatile i32 38, i32 addrspace(1)* undef
  br label %divergent.ret0

divergent.ret0:
  store volatile i32 11, i32 addrspace(3)* undef
  ret void

divergent.ret1:
  store volatile i32 42, i32 addrspace(3)* undef
  ret void

uniform.ret:
  store volatile i32 9, i32 addrspace(1)* undef
  ret void
}

; IR-LABEL: @uniform_complex_multi_ret_nest_in_divergent_triangle(
; IR: Flow1:                                            ; preds = %uniform.ret1, %uniform.multi.exit.region
; IR: %6 = phi i1 [ false, %uniform.ret1 ], [ true, %uniform.multi.exit.region ]
; IR: br i1 %6, label %uniform.if, label %Flow2

; IR: Flow:                                             ; preds = %uniform.then, %uniform.if
; IR: %7 = phi i1 [ %uniform.cond2, %uniform.then ], [ %uniform.cond1, %uniform.if ]
; IR: br i1 %7, label %uniform.endif, label %uniform.ret0

; IR: UnifiedReturnBlock:                               ; preds = %Flow3, %Flow2
; IR-NEXT: call void @llvm.amdgcn.end.cf(i64 %5)
; IR-NEXT: ret void
define amdgpu_kernel void @uniform_complex_multi_ret_nest_in_divergent_triangle(i32 %arg0) #0 {
entry:
  %id.x = tail call i32 @llvm.amdgcn.workitem.id.x()
  %divergent.cond0 = icmp eq i32 %id.x, 0
  br i1 %divergent.cond0, label %uniform.multi.exit.region, label %divergent.ret

uniform.multi.exit.region:
  %uniform.cond0 = icmp eq i32 %arg0, 4
  br i1 %uniform.cond0, label %uniform.if, label %uniform.ret1

uniform.if:
  %sgpr0 = load volatile i32, i32 addrspace(2)* undef
  %uniform.cond1 = icmp slt i32 %sgpr0, 1
  br i1 %uniform.cond1, label %uniform.then, label %uniform.endif

uniform.then:
  %sgpr1 = load volatile i32, i32 addrspace(2)* undef
  %uniform.cond2 = icmp sge i32 %sgpr1, 4
  store volatile i32 33, i32 addrspace(1)* undef
  br i1 %uniform.cond2, label %uniform.ret0, label %uniform.endif

uniform.endif:
  store volatile i32 38, i32 addrspace(1)* undef
  br label %uniform.ret0

uniform.ret0:
  store volatile i32 11, i32 addrspace(3)* undef
  ret void

uniform.ret1:
  store volatile i32 42, i32 addrspace(3)* undef
  ret void

divergent.ret:
  store volatile i32 9, i32 addrspace(1)* undef
  ret void
}

; IR-LABEL: @multi_divergent_unreachable_exit(
; IR: UnifiedUnreachableBlock:
; IR-NEXT: call void @llvm.amdgcn.unreachable()
; IR-NEXT: br label %UnifiedReturnBlock

; IR: UnifiedReturnBlock:
; IR-NEXT: call void @llvm.amdgcn.end.cf(i64
; IR-NEXT: ret void
define amdgpu_kernel void @multi_divergent_unreachable_exit() #0 {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  switch i32 %tmp, label %bb3 [
    i32 2, label %bb1
    i32 0, label %bb2
  ]

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  unreachable

bb3:                                              ; preds = %bb
  switch i32 undef, label %bb5 [
    i32 2, label %bb4
  ]

bb4:                                              ; preds = %bb3
  ret void

bb5:                                              ; preds = %bb3
  unreachable
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
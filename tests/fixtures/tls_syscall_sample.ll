; Minimal IR snippet for concurrency pass regression tests.
target triple = "x86_64-unknown-linux-gnu"

@virtual_fs = global [4096 x i8] zeroinitializer

define i32 @sample_tls_and_syscall() {
entry:
  %off = add i32 0, 16
  %v = call i8 @"__readfsbyte"(i32 %off)
  call void @"__writefsbyte"(i32 %off, i8 %v)
  %rc = call i64 @"syscall"(i64 202, i64 0, i32 0)
  ret i32 0
}

declare i8 @"__readfsbyte"(i32)
declare void @"__writefsbyte"(i32, i8)
declare i64 @"syscall"(i64, i64, i32)

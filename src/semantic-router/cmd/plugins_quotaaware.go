package main

// Registers the "quota_aware" decision algorithm (see
// pkg/selection/quotaaware) with the Router at startup. This is the only
// change this fork makes to any file the Router's own cmd package ships;
// everything else is consumed unmodified as a library.
import (
	_ "github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/quotaaware"
)

# Study 27a — deviations from the frozen specification

- ZA12 failed as specified. It required that retiming a history (any linear extension of its dependence order) leave the reached set
  and the *total order* of activation unchanged. The reached set was unchanged in 1500 of 1500 retimings, but the total order differed
  in 474: two members reached through independent events can be reached in either order. The specification asked for more than the
  trace determines. Restated: the reached set and the activating event of every member are invariant (every candidate activating event
  of a member contains that member, so these events are pairwise dependent and their order is fixed by the trace). The restated check
  is in check_study27a.py, with the original outcome printed alongside.
- No other change. All other checks ran as specified.

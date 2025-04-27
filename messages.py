# implementation
MSG_UNIMPLEMENTED = "Function not implemented!"
MSG_UNSOLVED = "Problem is unsolvable"

# graph
MSG_INVALID_GRAPH_SPEC = "invalid graph spec"
MSG_INVALID_GRAPH_TOPO = "invalid graph topology error"

# Comm
MSG_INVALID_GOALS = "The specified DOFs are not achievable with a single joint. consider adding an extra stage inbetween and distribute the target DOFs to the two joints. (Hint: no single joint can enable 3 translations)"
MSG_VALID_GOALS = "Target DOFs are achievable."
MSG_GOALS_DONE = "The targeted DOFs are achieved."
MSG_INVALID_INPUT = "Invalid input."
MSG_SIZE_MISMATCH = "Input array has an invalid size."
MSG_INPUT_NOT_SCREW = "Input is not screw vector(s)."


MSG_VALID_GOALS = "Target DOFs are achievable."
MSG_GOALS_DONE = "The targeted DOFs are achieved."
MSG_NEED_MORE_CONS_AXES = "Need more axes (different directions) from these constraint spaces."
MSG_NEED_MORE_CONS_SPACES = "Need more axes (different positions) from these constraint spaces."
MSG_NO_PLACEHOLDER = "No placeholder joint exists in model"
MSG_SHOW_NEEDED = "Showing needed constreaint space."
MSG_NO_OC_ROD = "No over-constraining rods found."
MSG_HAS_OC_ROD = "Over-constraining rod found! Please delete the highlighted rod(s) to continue."
MSG_ROD_COMPLETE = "Flexural rod design is complete."
MSG_ROD_INCOMPLETE = "Flexural rod design is incomplete. Please go back to Step 2."
MSG_ROD_SPACE_COMPLETE = "Flexural rod design complete. Please move forward to step 3."
MSG_ROD_SPACE_INCOMPLETE = "Need more flexural rods."


def MSG_OVER_CONS(flexures):

    count = len(flexures)
    if(count == 0):
        msg = "No over-constraining flexures found."
    else:
        ids = ", ".join([str(f.id) for f in flexures])
        if(count == 1):
            msg = "Flexure #%s is an overconstraining element. Please remove it." % ids
        else:
            msg = "Flexure #%s are overconstraining elements. Please remove them." % ids
        
    return msg

def MSG_CHECKLIST(tarAxisDeg, tarSpanDeg, curAxisDeg, curSpanDeg):
    
    axisComplete = tarAxisDeg == curAxisDeg
    spanComplete = tarSpanDeg == curSpanDeg
    
    axisCheck = "(v)" if axisComplete else "(x)"
    spanCheck = "(v)" if spanComplete else "(x)"

    _tarSpanDeg, _curSpanDeg = tarSpanDeg, curSpanDeg
    if(tarSpanDeg == 0):
        _tarSpanDeg += 1

    msgs = []
    msgs += ["%s required unique wire axis directions: %d/%d" % (axisCheck, curAxisDeg, tarAxisDeg)]
    msgs += ["%s required unique wire positions: %d/%d" % (spanCheck, _curSpanDeg, _tarSpanDeg)]
    
    msg = '\n'.join(msgs)

    return msg
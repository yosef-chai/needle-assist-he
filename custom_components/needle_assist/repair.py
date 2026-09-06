"""What the sentence says a call should have been, before anything runs it.

Everything :meth:`executor.Executor.execute` does between reading the model's
answer and calling a Home Assistant service: decode ``(tool, action)`` to the
virtual id, let the Hebrew verb settle which behaviour it really is, fill the
slots the sentence owns, and drop the arguments it does not support. None of
that needs a running Home Assistant, which is the whole reason this file can
exist.

It lives apart from :mod:`executor` because there were two copies of it. The
evaluation harness carried its own, under a comment reading "what
``executor.execute`` does before it runs anything, in its order" - a hand-kept
transcription of two hundred lines. It drifted three times in one session:
``extract_message`` was never called there, ``get_weather`` ignored its own
``day_offset``, and the simulated house had no ``button`` domain at all though
the corpus presses five of them 121 times. Every one of those read as a model
failure and was a harness failure.

So the harness and the household now run the same function, and a repair
measured on the benchmark is the repair that ships.

The order is load-bearing throughout, and every step carries the corpus
measurement that put it there. The invariant the file is built on is **settle
the behaviour, then fill its slots**; violating it costs the row twice, once
for the action and once for the argument it had nowhere to put.

Returns ``("", args)`` for a name in neither catalogue. The arguments come back
anyway, so the caller can say what it was asked to do.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from . import direction, slot_match, tool_router
from .const import (
    ACTION_ARG,
    ACTIONS,
    BOOLEAN_SLOT,
    CALL_OF,
    DURATION_TOOLS,
    HELPER_IS_A_TIMER,
    INTEGER_SETTING,
    MEDIA_TOOLS,
    NUMBER_SLOT,
    SETTING_SLOT,
    STEP_SOURCE,
    TOOL_ARGS,
    VIRTUAL_OF,
)

_LOGGER = logging.getLogger(__name__)


def settle(name: str, arguments: dict[str, Any] | None,
           utterance: str = "") -> tuple[str, dict[str, Any]]:
    """The virtual id and the arguments this call should carry.

    ``name`` is what the model emitted - a v11 tool name, or a v10 virtual id
    from a household's own older fine-tune. ``utterance`` is the clause, not
    the sentence: :mod:`clause_split` cuts it before the model ever sees
    it, and every rule below is measured per clause.
    """
    # A blank string is the model failing to fill a slot, not a value it
    # chose. Gold carries none in 60k corpus calls, and the row pays twice
    # for one: an invented argument and a lost exact match. Eight rows of
    # the frozen benchmark, every one a `name` on a timer or a button.
    args = {k: v for k, v in (arguments or {}).items()
            if not (isinstance(v, str) and not v.strip())}

    # What the model emits is ``(tool, action)``; what everything below
    # this line speaks is the **virtual id** - the per-service name the
    # catalogue used before v11. Decoding here rather than threading the
    # pair through is what let the catalogue collapse from 42 tools to 20
    # without touching the direction guard, the service map, the routine
    # sibling rule, the timer correction or any of their measurements.
    #
    # A name that is already a virtual id passes through unchanged, so a
    # household still running v10 weights against this component keeps
    # working - and so does every test written before the rewrite.
    action = args.pop(ACTION_ARG, None)
    tool = VIRTUAL_OF.get(
        (name, action if isinstance(action, str) else None))
    if tool is None:
        tool = name if name in CALL_OF else ""
    if not tool:
        # Almost always one thing: weights trained against a different
        # catalogue. v11 replaced 42 per-service tools with 20 per-domain
        # ones, so a household running their own v10 fine-tune emits
        # `light_turn_off` where this expects `light_control{shut}` - and
        # `light_turn_off` still decodes, which is why the fallback above
        # exists. What reaches here is a name from neither catalogue.
        _LOGGER.warning(
            "%r is not a tool this catalogue has%s - if `weights_path` "
            "points at a fine-tune of your own, it was trained against a "
            "different tool set", name,
            f" (action {action!r})" if action else "")
        return "", args

    # A helper toggle asked about a countdown is a timer command. Done
    # before the direction guard rather than after, so the guard settles
    # start against cancel on the tool this leaves behind: the toggle only
    # carries on or off, and "עצור את הטיימר" arrives as *turn_on*.
    # See const.HELPER_IS_A_TIMER for the corpus measurement.
    if (utterance and tool in HELPER_IS_A_TIMER
            and tool_router.names_a_timer(utterance)):
        _LOGGER.debug("%r is about a timer, not a helper toggle", utterance)
        tool = HELPER_IS_A_TIMER[tool]

    # The noun says which family; the verb below says which end of it. The
    # model disagrees with the noun far more often than the noun is wrong:
    # 19,410 agree and 7 disagree over the corpus, and six of the seven are a
    # word of speech noise glued to its neighbour. See `direction.FAMILY_ANCHOR`.
    #
    # **First, because it answers with the family's anchor and so throws the
    # direction away.** `family_named` returns `FAMILY_ANCHOR[named]` - for the
    # cameras that is `camera_turn_on` - so a sentence saying סגור that the
    # model answered with a cover came out of this switching the camera *on*.
    # The evaluation harness ran the two the other way round and read those
    # rows as correct; the household got them wrong. Eleven rows of the frozen
    # benchmark, eight of them cameras, and it is what made the two copies of
    # this chain worth merging.
    if utterance:
        settled = direction.family_named(
            tool, tool_router.family_named(utterance))
        if settled != tool:
            _LOGGER.debug("%r names a %s, not a %s", utterance,
                          direction.family_of(settled), tool)
            tool = settled

    # Lock or unlock, open or close, on or off. The Hebrew verb settles it and
    # the model does not always agree with the verb - see `direction`, where
    # the signal is measured at 1337 right and 0 wrong against gold. After the
    # family, so it settles the pair the noun just chose.
    if utterance and (settled := direction.settle(tool, utterance)) != tool:
        _LOGGER.debug("the sentence says %s, not %s", settled, tool)
        tool = settled

    # And the one family a sentence names outright that `family_named` cannot
    # read: an automation is named after what it does, so "האוטומציה תריסים
    # בבוקר" names two families and settles nothing. 1,172 agree, 3 disagree;
    # see `direction.ROUTINE_NOUNS`.
    if utterance:
        settled = direction.settle_routine_noun(tool, utterance)
        if settled != tool:
            _LOGGER.debug("%r names a routine, not a %s", utterance, tool)
            tool = settled

    # A toggle has no opposite, so the pair above cannot reach it - and
    # v11 made that a live problem: all three of on, off and toggle are
    # values of one enum now, where the router used to rank the toggle
    # third and a tight shortlist usually cut it. 335 agree and 0 disagree
    # on the rows whose gold really is a toggle; see `direction.TOGGLES`.
    if utterance and (settled := direction.settle_toggle(
            tool, utterance,
            slot_match.level_from(utterance) is not None)) != tool:
        _LOGGER.debug("%r says %s rather than a toggle", utterance, settled)
        tool = settled

    # A plug that is a speaker. A transport verb the model answered with a
    # switch is a media command - the television really is a plug in most
    # of this corpus, and a plug has no pause. `settle_media` below says
    # which of the eight. See `direction.TRANSPORT_IS_MEDIA`.
    if (utterance and tool in direction.TRANSPORT_IS_MEDIA
            and tool_router.names_a_transport(utterance)):
        _LOGGER.debug("%r is a transport verb, not a %s", utterance, tool)
        tool = "media_pause"

    # A door that is a blind. "תפתח את דלת החניה" is the garage, and דלת
    # alone is what a lock has, so the model answers `lock_control` and a
    # household that asked for the garage gets a bolt. All 147 corpus rows
    # naming one of the compound cover nouns are cover rows; see
    # `direction.NAMES_A_COVER`. After the toggle so a `switch_toggle` has
    # already become one side of its pair, and before the rest so `settle`
    # still says open or closed.
    if (utterance and tool in direction.NAMES_A_COVER
            and slot_match.names_a_compound_cover(utterance)):
        _LOGGER.debug("%r names a cover, not a %s", utterance, tool)
        tool = direction.NAMES_A_COVER[tool]
        tool = direction.settle(tool, utterance)

    # The number the model dropped, and the number it got wrong. "שנה את
    # הטמפרטורה ל20 מעלות" comes back as `climate_control{off}` with no
    # argument, and an air conditioner switches off when somebody asked
    # for twenty degrees.
    #
    # This used to fire only where the model left the slot empty. It is an
    # override now, because the sentence was measured to be right about
    # this whenever it speaks at all - 485 agree and 2 disagree over the
    # corpus, both of them "אשרים" - and a model that emits 23 for a
    # sentence saying 19 is the commoner failure of the two. Same rule as
    # the room, the floor, the name and the colour: the sentence decides.
    # See `slot_match.temperature_from`.
    if (utterance and tool in direction.CLIMATE_BEHAVIOURS
            and not args.get("temperature_step")
            and (degrees := slot_match.temperature_from(utterance))):
        _LOGGER.debug("the sentence says %s degrees", degrees)
        args["temperature"] = degrees

    # A climate call carrying a temperature is a call to set one. Not a
    # word in the sentence - the argument the model itself emitted, which
    # only one of the five behaviours has anywhere to put. Measured on
    # predictions rather than on gold: 160 agree, 1 disagrees. Unless the
    # clause asked for a fan speed or an hvac mode, in which case it asked
    # for that. See `direction.settle_climate`.
    if tool in direction.CLIMATE_BEHAVIOURS:
        settled = direction.settle_climate(
            tool, args, utterance,
            slot_match.mode_slot(utterance),
            None if (slot_match.setting_from(utterance, "fan_mode")
                     or slot_match.temperature_from(utterance))
            else slot_match.hvac_target(utterance),
            # The same test `settle_steps` gets below: no temperature said
            # and no digit either. A `temperature` on such a clause was
            # invented, and it must not outrank a mode that was spoken.
            not slot_match.unsupported(utterance, "temperature"))
        if settled != tool:
            _LOGGER.debug("%s carries a temperature, so it is %s",
                          tool, settled)
            tool = settled

    # And the two tools whose behaviours the sentence separates without
    # ever being wrong. An allow-list: the general form of this rule was
    # measured over the whole corpus at 647 disagreements and rejected.
    # See `direction.HINT_DECIDED`.
    if utterance and (name in direction.HINT_DECIDED
                      or CALL_OF.get(tool, ("",))[0] in direction.HINT_DECIDED):
        siblings = list(ACTIONS.get(CALL_OF[tool][0], {}).values())
        if (settled := direction.settle_action(tool, siblings, utterance)) != tool:
            _LOGGER.debug("the sentence says %s, not %s", settled, tool)
            tool = settled

    # Which of seven, for a countdown. The one tool the model does not do
    # at all - it collapsed onto two of the seven - and the one whose
    # behaviours the hint tables cannot separate either. 1262 agree and 6
    # disagree; see `direction.settle_timer`. It is also where a transport
    # verb aimed at a countdown lands back in the right family - see
    # `direction.TRANSPORT_IS_A_TIMER`.
    #
    # **After the allow-list, not before it.** `settle_action` reads the
    # router's hint lists, where עצור sits under cancel; "לעצור **רגע** את
    # הספירה" is a pause, and only `_T_PAUSE` carries that phrase. With
    # this block first, the allow-list then overruled it on eleven such
    # rows. Letting the specialist speak last costs nothing anywhere else
    # and gains 118 rows over the corpus - see `direction.HINT_DECIDED`.
    if utterance and tool in direction.SETTLES_A_TIMER:
        settled = direction.settle_timer(
            tool, args, utterance,
            tool_router.looks_like_question(utterance),
            tool_router.names_a_timer(utterance))
        if settled != tool:
            _LOGGER.debug("the sentence says %s, not %s", settled, tool)
            tool = settled

    # The clause that names its own behaviour outright: a fan asked to
    # turn, a cover asked to halt, a speaker asked to stop rather than
    # pause, a cover asked for a percentage. Four families
    # `settle_action`'s allow-list cannot take; see
    # `direction.settle_named` for the evidence.
    #
    # Before the numbers below, because a cover promoted to a placement
    # has to be one by the time `position` is filled in.
    if utterance:
        settled = direction.settle_named(
            tool, utterance,
            slot_match.level_from(utterance) is not None,
            slot_match.setting_from(utterance, "color_name") is not None)
        if settled != tool:
            _LOGGER.debug("the clause names %s, not %s", settled, tool)
            tool = settled

    # Which of eight, for a speaker. 2020 agree, 1 disagree; see
    # `direction.settle_media`. **Before** the slot fills below, not
    # after: they are keyed on the behaviour, so a call promoted here to
    # `media_mute` had already been filled as whatever it was, and
    # `is_volume_muted` is not an argument of that. The same defect
    # `settle_climate` had, one family over - settle the behaviour, then
    # fill its slots. `evaluate.py` runs it in this place too.
    if utterance and (tool in direction.MEDIA_BEHAVIOURS
                      or tool == "music_play"):
        settled = direction.settle_media(
            tool, utterance, slot_match.names_a_level(utterance),
            slot_match.a_plain_request(utterance),
            slot_match.level_from(utterance) is not None)
        if settled != tool:
            _LOGGER.debug("the sentence says %s, not %s", settled, tool)
            tool = settled

    # And the other numbers Hebrew says out loud, read *after* every rule
    # that can still change which behaviour this is - unlike the
    # temperature above, which has to come first because
    # `settle_climate` promotes on it.
    #
    # Half of every number in the corpus is a word rather than a digit -
    # "שבעים וחמישה אחוז", "חצי", "רבע שעה" - and nothing here could read
    # one until `hebrew_numbers`. 1,175 agree and 3 disagree on the
    # percentage; see `const.NUMBER_SLOT`.
    if utterance and "day_offset" in TOOL_ARGS.get(tool, frozenset()):
        # Truthy, not `is not None`: today is spelled by saying nothing.
        if (day := slot_match.day_offset_from(utterance)):
            args["day_offset"] = day
        else:
            args.pop("day_offset", None)

    if (utterance and (slot := NUMBER_SLOT.get(tool))
            and (said := slot_match.level_from(utterance)) is not None):
        args[slot] = said

    # And the yes-or-no, read the same way and in the same place. A fan
    # asked to turn and a speaker asked to be silenced both carry one, and
    # the sentence always says which way; see `const.BOOLEAN_SLOT`.
    if utterance and (slot := BOOLEAN_SLOT.get(tool)):
        args[slot] = slot_match.switch_from(utterance, slot)

    # And the countdown. 553 exactly right, 7 wrong, and silent on every
    # timer row whose gold carries no duration - which is what keeps a
    # pause from acquiring one. The units are replaced together rather
    # than merged, because a model that said forty minutes for a sentence
    # saying two hours must not leave the forty behind.
    # See `const.DURATION_TOOLS`.
    if (utterance and tool in DURATION_TOOLS
            and (said_time := slot_match.duration_from(utterance))):
        for unit in ("hours", "minutes", "seconds"):
            args.pop(unit, None)
        args.update(said_time)

    # And the mirror of every rule above: a slot the sentence owns and
    # does not name is a slot the model invented. 459 invented arguments
    # in one release run; see `slot_match.SENTENCE_OWNS` for the nine
    # slots this is measured safe on and the four it is not.
    if utterance:
        for slot in [k for k in args if slot_match.unsupported(utterance, k)]:
            _LOGGER.debug("the sentence does not support %s=%r",
                          slot, args[slot])
            args.pop(slot, None)

    # Which step slots this behaviour declares and the sentence has left
    # room for. See `const.STEP_SOURCE`.
    def _fillable_steps(name: str, said: str) -> tuple[str, ...]:
        takes = TOOL_ARGS.get(name, frozenset())
        return tuple(slot for slot, source in STEP_SOURCE.items()
                     if slot in takes
                     and slot_match.unsupported(said, source))

    # The same verb also settles which way a relative argument points, and
    # `_service_data` below adds those to the device's current reading - so
    # a wrong sign moves the thermostat away from what was asked instead of
    # towards it. 1222 right and 1 wrong; see `direction`.
    if utterance:
        args = direction.settle_steps(
            args, utterance, _fillable_steps(tool, utterance))

    # "Play" and "play *this*" are one verb apart in Hebrew, and which one
    # was meant is decided by whether a name follows - which the sentence
    # settles and the model has to guess. So when the model reaches for
    # some other media tool about a sentence that named something
    # specific, the sentence wins.
    #
    # This is the same rule the rest of this module runs on, and it is
    # safe here for the same reason: it never overrides which *domain* was
    # chosen. The model has already decided the utterance is about audio -
    # "תפעיל את השואב" gets vacuum_start and is never seen here - so all
    # that is being corrected is which media tool inside that decision. It
    # also makes the tool work before any model knows it exists, which is
    # what a household running the previous weights has.
    #
    # Two strengths of evidence, because they carry different risks. A
    # model that already said *play* has only the "what" left to get
    # wrong, so any title is enough. A model that said something else -
    # set the volume, pause - is being overruled on the verb too, so the
    # sentence has to have named the **kind** as well ("האלבום", "פלייליסט")
    # and no number: "שים את השיר על שישים" is a volume and says so.
    #
    # Measured over both splits, on the 663 media rows where the sentence
    # names a kind and a title and no number: 661 are music_play and the
    # two that are not are the same glued-ו artifact the narrow rule
    # already mishandles today, so the widening breaks nothing new. On the
    # 123 music rows of the held-out set it takes tool-set from 71.5% to
    # **88.6%** and argument F1 from 72.9% to 89.7% - twenty-one rows,
    # most of them "ערבב את האלבום", shuffle, answered with
    # media_set_volume.
    if tool in MEDIA_TOOLS and utterance:
        request = slot_match.extract_music(utterance)
        if request is not None and (
                tool == "media_play"
                or (request.media_type
                    and not slot_match.names_a_level(
                        utterance, request.media_id))):
            _LOGGER.debug("the sentence names something to play; using "
                          "music_play instead of %s", tool)
            tool = "music_play"

    # Announce it to the house, or send it to a phone. The two take the same
    # single argument and the model confuses them in both directions; the verb
    # never does. 330 agree, 0 disagree; see `direction.settle_audience`.
    #
    # And the case where it confused them with something else entirely: the
    # speakers were heard and the message was not, so an announcement came
    # back as a transport command and, once, as every lock in the house. Asked
    # only where the model answered outside the pair, so the rule above keeps
    # the cases it was measured on. 248 agree, 0 disagree; see
    # `direction.announces`.
    if utterance and tool not in direction.AUDIENCE and (
            announced := direction.announces(
                utterance, slot_match.extract_message(utterance) is not None)):
        _LOGGER.debug("%r announces something; using %s instead of %s",
                      utterance, announced, tool)
        tool = announced
    if utterance and tool in direction.AUDIENCE:
        settled = direction.settle_audience(
            tool, utterance, slot_match.extract_message(utterance) is not None)
        if settled != tool:
            _LOGGER.debug("the verb in %r says %s, not %s", utterance,
                          settled, tool)
            tool = settled

    # And the mirror of that upgrade: a `music_play` about a sentence
    # that named nothing to play, and named one transport verb, is a
    # transport command. 383 agree, 0 disagree; see `direction`.
    if tool == "music_play" and utterance:
        settled = direction.settle_transport(
            tool, utterance, slot_match.extract_music(utterance) is not None)
        if settled != tool:
            _LOGGER.debug("nothing to play in %r; using %s", utterance, settled)
            tool = settled

    # Slots the sentence names outright, which is where `_service_data` read
    # them until this file existed. They are not a translation of the model's
    # answer, they replace it: the speaker said "שקט" or "גבוה", and which of
    # the enum's values that is needs no model - 474 right and 0 wrong, and
    # 3,354 right and 0 wrong over the whole corpus with the colour included.
    # A tuple since v11, because one behaviour can carry more than one.
    #
    # Last, because every rule above can still change which behaviour this is,
    # and the slots are keyed on it - a call promoted to `music_play` two lines
    # up takes `media_type` here rather than keeping whatever it had.
    for slot in SETTING_SLOT.get(tool, ()):
        if not utterance or slot not in TOOL_ARGS.get(tool, frozenset()):
            continue
        # `value`, not `said`: the level fill above binds that name to an int,
        # and these tables are strings.
        if (value := slot_match.setting_from(utterance, slot)) is not None:
            # The tables are strings throughout; the schema is not.
            # See `const.INTEGER_SETTING`.
            args[slot] = int(value) if slot in INTEGER_SETTING else value

    return tool, args


#: Behaviours a failed generation may be rebuilt into. Read-only, every one of
#: them: recovering an *actuation* from a generation that broke would move a
#: device on the strength of a sentence nobody finished reading.
RECOVERABLE: Final = frozenset(("get_state", "get_weather", "get_datetime"))

#: The two speaker tools. They take a volume the same way, so a shortlist
#: holding both is two tools wide and still not a choice about *this*. See the
#: volume branch of :func:`recover`.
_SPEAKERS: Final = frozenset(("media_control", "music_play"))


def recover(utterance: str) -> dict[str, Any] | None:
    r"""The call a *failed* generation should have produced, or ``None``.

    An engine failure is not a refusal, and until this existed the two were
    answered the same way. Both arrive as an empty call list; ``needle_runner
    .failed`` tells them apart, and what it found was a real and specific
    defect rather than a token budget: the model emits Hebrew as ``\uXXXX``
    escape sequences - which is how every Hebrew argument value was written in
    its own training targets - and once it opens one inside a constrained slot
    it cannot close it. ``{"domain":"med\u05d4...`` never matches an enum, and
    the decoder runs to the budget. Forty-one rows of the frozen benchmark;
    raising the budget from 192 to 512 recovers three of them, because the
    budget was never the problem.

    Two conditions, and both are the scope rather than a heuristic:

    * **the router offered exactly one tool**, so there was nothing for the
      model to choose and naming it is not a guess;
    * **that tool cannot move anything**, so a mistake here answers a question
      nobody asked instead of opening a blind.

    Measured per clause over the whole corpus, on the rows the pre-inference
    gates let through: **2,252 agree and 2 disagree**. Widened to the whole
    shortlist it reads 2,269/10, and widened to the message tools 2,579/41 -
    the forty-one being ``notify_send`` answered with ``broadcast``, which is a
    corpus disagreement rather than a reading error. Of the forty-one rows that
    actually fail this way, **none** of the off-topic ones has a single-tool
    shortlist: an off-topic sentence gets a wide one, which is the property
    that makes the narrow scope safe rather than merely cautious.

    The arguments come from the same resolvers the executor uses, so a
    recovered call is filled exactly as a generated one would have been.

    **Two widenings measured and declined**, so nobody measures them again.
    Each would have reached one more of the ten rows a derailed generation
    still cost on the frozen benchmark, and each fails the bar:

    * *a countdown* - the shortlist is only the timer and the clause says how
      long - **439 agree, 209 disagree**. The condition is right about the
      tool and says nothing about the behaviour: "תוסיף עוד חמש דקות לטיימר"
      is an `add` and "תוריד 5 דקות מהטיימר" a `less`, and both name a
      duration. `tool_router.asks_for_a_reminder` answers a different question
      - may this sentence reach the model - and is sound for that one.
    * *a placement* - a cover or a valve on the shortlist and a percentage to
      place it at - **384 agree, 2 disagree**: השסטום, a valve whose noun the
      noise turned into a cover, and a countdown shortened by a quarter of an
      hour, which `percent_from` reads as twenty-five per cent. Two is not
      zero, and here it would move the wrong machine.

    And a note on how those numbers are taken, because it changed one verdict.
    The message pair's first reading said 326/20 and was measured over the
    whole corpus: 18 of the 20 are "תשלח מייל לבוס שאני חולה", an email this
    house has no tool for, and `looks_off_topic` refuses it before the model is
    ever asked, so this function is never handed it. Measured where the
    paragraph above says - on the rows the gates let through - the same
    widening reads 325/0 and was adopted. **The scope is part of the
    measurement**, and the two candidates declined here were re-measured inside
    it before being declined.
    """
    names = tool_router.select_tool_names(utterance, tool_router.MAX_TOOLS)
    virtual = tool_router.select_virtual_names(utterance, 1)
    tool = virtual[0] if virtual else (names[0] if names else "")

    # The one shortlist wider than a single tool that is still not a choice:
    # `broadcast` and `notify_send` are the whole of it, they take the same
    # single argument, the executor reads that argument off the sentence for
    # either - and the verb settles which, at 330 agreements and none against.
    # Twelve of the forty-one failures are here, every one an announcement
    # whose Hebrew message is what derailed the generation in the first place.
    # Offered *alongside* other tools too, which is the widening round seven
    # made. "תשדר ברמקולים שיוצאים בעוד חמש דקות" shortlists both speaker
    # tools as well, so the exact-pair test this used to be declined it and
    # the household got nothing at all.
    #
    # The first reading of the widening said 326 agree / 20 disagree and was
    # measuring rows this function never sees: 18 of the 20 are "תשלח מייל
    # לבוס שאני חולה", an email this house has no tool for, and
    # `looks_off_topic` refuses it before the model is ever asked. Measured
    # where the docstring says - per clause over the v11 corpus, on the rows
    # the pre-inference gates let through - the pair reads **325 agree, 0
    # disagree**, and the broadcast half alone 99/0.
    if set(direction.AUDIENCE) <= set(names):
        # The verb has to actually speak. Left to fall back on the router's
        # own ranking inside the pair it reads 39 rows wrong, all of them
        # `תשלח הודעה לכולם` - a notification *to everyone*, which is the one
        # phrasing where the audience and the verb point opposite ways.
        if (slot_match.extract_message(utterance) is not None
                and (settled := direction.settle_audience("", utterance, True))):
            return {"name": settled, "arguments": {}}
        if set(names) == set(direction.AUDIENCE):
            # Nothing else was on the shortlist, so there is nothing below
            # this worth trying. Where other tools were offered, there is.
            return None

    # And the third thing the sentence names outright: a routine. The noun
    # says the family and the verb says which end of it, at 1,172 agreements
    # and 3 disagreements over the corpus - the three being the speech noise
    # `direction.DISCRIMINATING` already names, where the noun is read and the
    # negative verb is not. It speaks on **none** of the 200 off-topic rows
    # that get past the gates, which is the same property that makes the
    # message pair safe. All five automation rows that fail this way are engine
    # failures rather than refusals: the model derails on the escape sequences
    # and the sentence has said the answer outright.
    if (settled := direction.settle_routine_noun("", utterance)):
        if settled not in CALL_OF:
            return None
        wire, action = CALL_OF[settled]
        return {"name": wire,
                "arguments": {ACTION_ARG: action} if action else {}}

    # The fourth thing the sentence names outright, and the first that moves
    # something. "תכוון את הקול בחדר השינה ל70 אחוז", "תנמיך עוד קצת את
    # העוצמה בבלקון" - the escape sequences derail the generation, the
    # household gets nothing, and the clause has said the behaviour, the
    # device and the number. Five of the ten rows a derailed generation
    # actually costs on the frozen benchmark are this one sentence shape.
    #
    # Two conditions, and they are what keep an actuation to the same standard
    # as the read-only ones above: a speaker is on the shortlist at all, and
    # the clause **names the volume** rather than merely carrying a number.
    # `direction.names_the_volume` is where that is measured - 391 agree and
    # none against, where reading the bare value disagrees three times and
    # reading the verb everywhere disagrees seven. `repair.settle` fills the
    # value afterwards from `NUMBER_SLOT`, exactly as for a generated call.
    if (_SPEAKERS & set(names)) and direction.names_the_volume(
            utterance, set(names) <= _SPEAKERS):
        wire, action = CALL_OF["media_set_volume"]
        return {"name": wire, "arguments": {ACTION_ARG: action}}

    if len(names) != 1 or tool not in RECOVERABLE:
        return None
    if tool == "get_state":
        # `domain` is required, and a question this cannot type is a question
        # this cannot answer. 1,467 right and 2 wrong over the corpus; see
        # `tool_router.query_domain`.
        if not (domain := tool_router.query_domain(utterance)):
            return None
        args: dict[str, Any] = {"domain": domain}
        if (wanted := slot_match.state_filter(utterance)):
            args["state"] = wanted
        return {"name": "get_state", "arguments": args}
    # The clock and the weather declare nothing required; `repair.settle` fills
    # `day_offset` and the executor answers from the house.
    return {"name": tool, "arguments": {}}

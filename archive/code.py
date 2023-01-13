#ce, c1, ce, c2, ce, c4, ce, c5 = st.columns([0.07, 1, 0.07, 1, 0.07, 1, 0.07, 1])
    #with c1:
        #start_button = st.button("Start DAQ")
    #    d1 = st.date_input("Start date", datetime.datetime.now())
    #    d2 = st.date_input("Stop date", datetime.datetime.now())

    #with c2:
    #    test = st.empty()
    #    start = "00:00"
    #    end = "23:59"
    #    times = []
    #    start = now = datetime.datetime.strptime(start, "%H:%M")
    #    end = datetime.datetime.strptime(end, "%H:%M")
    #    while now != end:
    #        times.append(str(now.strftime("%H:%M")))
    #        now += datetime.timedelta(minutes=1)
    #    times.append(end.strftime("%H:%M"))
    #    t1 = st.selectbox('Start time:',times)
    #    t2 = st.selectbox('Stop time:',times)

    
#if not daq_button:
#    st.stop()
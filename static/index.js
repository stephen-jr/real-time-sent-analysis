function eventSource(){
    const a = new EventSource('http://localhost:5000/stream')
    a.onmessage = (stream) => {
        let { data } = stream;
        data = JSON.parse(data)
        if(data.execution)console.log(data)
        else a.close()
    }
}
setTimeout(eventSource, 3000)
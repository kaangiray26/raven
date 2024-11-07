document.addEventListener("DOMContentLoaded", async () => {
    window.raven = new Raven();
    console.log("Raven loaded.");
});

class Raven {
    constructor() {
        this.store = {
            offset: 0,
        };
        this.config = {};
        this.last_update = null;
        this.ws = new WebSocket("ws://localhost:8000/ws");

        // Custom events
        this.onpdf = () => {};
        this.onusage = () => {};
        this.onjobs = () => {};
        this.onanswers = () => {};
        this.onkeywords = () => {};
        this.onaddpaper = () => {};
        this.ononsimilar = () => {};
        this.onsearchpapers = () => {};
        this.onpaperkeywords = () => {};

        // Event listeners for the websocket
        this.ws.onopen = () => {
            console.log("Connected to the server.");
        };
        this.ws.onclose = () => {
            console.log("Connection to the server closed.");
        };
        this.ws.onmessage = (message) => {
            const response = JSON.parse(message.data);
            let render = null;
            switch (response.type) {
                case "keywords":
                    this.onkeywords(response.data);
                    break;

                case "paper_keywords":
                    this.onpaperkeywords(response.data);
                    break;

                case "answers":
                    this.onanswers(response.data);
                    break;

                case "similar_papers":
                    this.onsimilarpapers(response.data);
                    break;

                case "add_paper":
                    this.onaddpaper(response.data);
                    break;

                case "usage":
                    this.onusage(response.data);
                    break;

                case "pdf":
                    this.onpdf(response.data);
                    break;

                case "log":
                    render = document.querySelector("pre");
                    if (!render) return;
                    render.innerHTML = response.data;
                    break;

                case "route":
                    window.location.href = response.data;
                    break;

                case "topic":
                    window.location.href = "/topics/" + response.data;
                    break;

                case "research":
                    window.location.href = "/research/" + response.data;
                    break;

                case "jobs":
                    this.onjobs(response.data);
                    break;

                case "search_papers":
                    this.onsearchpapers(response.data);
                    break;

                case "topic_papers":
                    console.log(
                        "Received papers:",
                        response.count,
                        response.template,
                    );
                    // Check if we have a papers field to update
                    const topic_papers =
                        document.querySelector("#topic-papers");
                    if (!topic_papers) return;

                    // Render the content
                    topic_papers.innerHTML += response.template;

                    // Save the paper count
                    this.store.offset += response.count;

                    // Update the ref
                    update("offset", `Offset: ${this.store.offset}`);
                    break;

                case "papers":
                    // Check if we have a papers field to update
                    const papers = document.querySelector("#papers");
                    if (!papers) return;

                    // Render the content
                    papers.innerHTML = response.content;
                    break;

                default:
                    console.log("Received:", response);
            }
        };
    }

    // Request to start a search job on a research question
    async research(query) {
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "research",
                query: query,
            }),
        );
    }

    // Request to search for related papers
    async search(query, criterion, sorted) {
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "search",
                query: query,
                criterion: criterion,
                sorted: sorted,
            }),
        );
    }

    // Request to get answers to a research question
    async generate_answers(query, source_urls){
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "generate_answers",
                query: query,
                source_urls: source_urls,
            }),
        );
    }

    // Request to add a new topic
    async add_topic(query) {
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "add_topic",
                query: query,
            }),
        );
    }

    // Request to add a new paper
    async add_paper(query) {
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "add_paper",
                query: query,
            }),
        );
    }

    // Request to update a topic
    async update_topic(id) {
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "update_topic",
                id: id,
            }),
        );
    }

    async get_keywords(offset) {
        return await fetch(
            "/keywords/list?" +
                new URLSearchParams({
                    offset: offset,
                }),
        ).then((res) => res.json());
    }

    async get_topic_papers(id, offset) {
        return await fetch(
            `/topics/${id}/papers?` +
                new URLSearchParams({
                    offset: offset,
                }),
        ).then((res) => res.json());
    }

    async add_to_favorites(source_url){
        // Send the request
        this.ws.send(
            JSON.stringify({
                type: "add_to_favorites",
                source_url: source_url,
            }),
        );
    }

    async get_favorite_papers(offset) {
        return await fetch(
            `/saved/papers?` +
                new URLSearchParams({
                    offset: offset,
                }),
        ).then((res) => res.json());
    }

    async get_related_papers(id) {
        // Get the ongoing jobs for the topic
        return await fetch(`/research/${id}/papers`).then((res) => res.text());
    }

    async show_message(title, message){
        // Get the error field
        const dialog = document.querySelector("#message-dialog");
        dialog.querySelector(".dialog-title").innerText = title;
        dialog.querySelector(".dialog-body").innerText = message;
        dialog.showModal();
    }

    async hide_message() {
        // Get the error field
        const dialog = document.querySelector("#message-dialog");
        dialog.close();
    }

    format_datetime(timestamp) {
        // Get the difference between the current time and the value
        const diff = new Date() - new Date(timestamp);

        // If the difference is less than 1 minute, return "just now"
        if (diff < 60 * 1000) {
            return "just now";
        }

        // If the difference is less than 1 hour, return the minutes
        if (diff < 60 * 60 * 1000) {
            return `${Math.floor(diff / (60 * 1000))} minutes ago`;
        }

        // If the difference is less than 1 day, return the hours
        if (diff < 24 * 60 * 60 * 1000) {
            return `${Math.floor(diff / (60 * 60 * 1000))} hours ago`;
        }

        // If the difference is less than 1 week, return the days
        if (diff < 7 * 24 * 60 * 60 * 1000) {
            return `${Math.floor(diff / (24 * 60 * 60 * 1000))} days ago`;
        }

        // If the difference is less than 1 month, return the weeks
        if (diff < 30 * 24 * 60 * 60 * 1000) {
            return `${Math.floor(diff / (7 * 24 * 60 * 60 * 1000))} weeks ago`;
        }

        // If the difference is less than 1 year, return the months
        if (diff < 365 * 24 * 60 * 60 * 1000) {
            return `${Math.floor(diff / (30 * 24 * 60 * 60 * 1000))} months ago`;
        }

        // If the difference is more than 1 year, return the years
        return `${Math.floor(diff / (365 * 24 * 60 * 60 * 1000))} years ago`;
    }

    format_datetimedelta(timestamp_1, timestamp_2) {
        // diff is in milliseconds, convert to seconds
        const diff = (new Date(timestamp_2) - new Date(timestamp_1)) / 1000;

        // If the seconds are less than 60, return the seconds
        if (diff < 60) {
            return `${diff.toFixed(2)} seconds`;
        }

        // Return in minutes
        return `${Math.floor(diff / 60)} minutes`;
    }

    format_long_date(timestamp){
        return new Date(timestamp).toLocaleDateString('en-GB', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }
}

async function update(ref, val) {
    document.querySelector(`*[ref="${ref}"]`).innerHTML = val;
}

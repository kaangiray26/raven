#!env/bin/python3
# -*- coding: utf-8 -*-

import time
import os
import json
import asyncio
import argparse
from base64 import b64decode, b64encode
from datetime import datetime
from hypercorn.asyncio import serve
import lib.waiter as waiter
from lib.worker import Worker
from hypercorn.config import Config
from quart import Quart, jsonify, render_template, request, websocket, redirect, send_from_directory
from lib.tables import get_db, row_factory, row_factory_custom, row_factory_list, rows as row_columns

app = Quart(__name__, template_folder=os.path.join("lib", "templates"), static_folder=os.path.join("lib", "static"))
app.config["TEMPLATES_AUTO_RELOAD"] = True


# Jinja2 filters
@app.template_filter()
def btoa(value):
    return b64encode(value.encode()).decode()


@app.template_filter()
def total_usage_cost(usage):
    return "${:.8f}".format(sum([float(v[1:]) for v in usage.values()]))


@app.template_filter()
def transform_related(value):
    return "/" + "/".join(value.split(":"))


@app.template_filter()
def boolean_ternary(value):
    return "True" if value else "False"


@app.template_filter()
def format_datetimedelta(start, end):
    diff = end - start

    if diff.seconds < 60:
        return f"{diff.seconds} seconds"

    return f"{diff.seconds // 60} minutes"


@app.template_filter()
def format_secondsdelta(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds"
    return f"{int(seconds) // 60} minutes"


@app.template_filter()
def format_datetime(value):
    # Get current time
    now = datetime.now()
    # Get the difference between the current time and the value
    diff = now - value

    # If the difference is less than 1 minute, return "just now"
    if diff.seconds < 60:
        return "just now"

    # If the difference is less than 1 hour, return the minutes
    if diff.seconds < (60 * 60):
        return f"{diff.seconds // 60} minutes ago"

    # If the difference is less than 1 day, return the hours
    if diff.seconds < (24 * 60 * 60):
        return f"{diff.seconds // 3600} hours ago"

    # If the difference is less than 1 week, return the days
    if diff.seconds < (7 * 24 * 60 * 60):
        return f"{diff.days} days ago"

    # If the difference is less than 1 month, return the weeks
    if diff.days < 30:
        return f"{diff.days // 7} weeks ago"

    # If the difference is less than 1 year, return the months
    if diff.days < 365:
        return f"{diff.days // 30} months ago"

    # If the difference is more than 1 year, return the years
    return f"{diff.days // 365} years ago"


@app.template_filter()
def format_datetime_string(value):
    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return format_datetime(dt)


@app.template_filter()
def format_dt_to_locale_string(dt):
    # Return as 01 January 2021
    return dt.strftime("%d %B %Y")


# Routes
@app.route("/404")
async def route_404():
    return await render_template("404.html")


@app.route("/")
async def route_index():
    return await render_template('index.html')


@app.route("/terms")
async def route_terms():
    return await render_template('terms.html')


@app.route("/privacy")
async def route_privacy():
    return await render_template('privacy.html')


@app.route("/topics")
async def route_topics():
    with get_db() as db:
        # Get all topics and manually added papers
        topics = db.sql("SELECT * FROM topics ORDER BY name ASC").df()
        papers = db.sql("SELECT * FROM papers WHERE related IS NULL ORDER BY added_at DESC").df()
    return await render_template('topics.html', topics=topics, papers=papers)


@app.route("/topics/<uuid:id>")
async def route_topic(id):
    print("Getting topic:", id)
    with get_db() as db:
        row = db.execute("SELECT * FROM topics WHERE id = ?", [id]).fetchone()
        paper_count = db.execute("SELECT COUNT(*) FROM papers WHERE related = ?", [f"topics:{id}"]).fetchone()
    if not row:
        return redirect("/404")

    topic = row_factory("topics", row)
    return await render_template("topic.html", topic=topic, paper_count=paper_count)


@app.route("/topics/<uuid:id>/papers")
async def route_topic_papers(id):
    # Get offset query param
    offset = request.args.get("offset", 0, int)

    # Get keywords from the database
    papers = await worker.get_topic_papers(id, offset)
    return json.dumps({
        "papers": papers
    }, default=str)


@app.route("/research")
async def route_research():
    with get_db() as db:
        research = db.execute("SELECT * FROM research ORDER BY updated_at DESC").df()
    return await render_template('researches.html', research=research)


@app.route("/research/<uuid:id>")
async def route_research_page(id):
    print("Getting research:", id)
    with get_db() as db:
        row = db.execute("SELECT * FROM research WHERE id = ?", [id]).fetchone()
    if not row:
        return redirect("/404")
    research = row_factory("research", row)
    return await render_template('research.html', research=research)


@app.route("/jobs")
async def route_jobs():
    return await render_template('jobs.html')


@app.route("/saved")
async def route_saved():
    return await render_template("saved.html")


@app.route("/saved/papers")
async def route_saved_papers():
    # Get offset query param
    offset = request.args.get("offset", 0, int)

    # Get keywords from the database
    papers = await worker.get_favorites(offset)
    return json.dumps({
        "papers": papers
    }, default=str)


@app.route("/papers")
async def route_papers():
    # Get the number of papers in the database
    with get_db() as db:
        count = db.sql("SELECT COUNT(*) FROM papers").fetchone()
    return await render_template('papers.html', count=count)


@app.route("/papers/<b64>")
async def route_paper(b64):
    # Decode the base64 id
    source_url = b64decode(b64).decode("utf-8")

    # Get the paper from the database
    # Also, check if the paper is saved
    with get_db() as db:
        row = db.execute("""
            SELECT
                p.*,
                CASE WHEN f.source_url IS NOT NULL THEN true ELSE false END AS is_favorite
            FROM papers p
            LEFT JOIN favorites f ON p.source_url = f.source_url
            WHERE p.source_url = ?
        """, [source_url]).fetchone()
    if not row:
        return redirect("/404")
    paper = row_factory_custom(row_columns["papers"] + ["is_favorite"], row)
    return await render_template('paper.html', paper=paper)


@app.route("/keywords")
async def route_keywords():
    return await render_template("keywords.html")


@app.route("/keywords/list")
async def get_keywords_list():
    # Get offset query param
    offset = request.args.get("offset", 0, int)

    # Get keywords from the database
    res = await worker.get_keywords(offset)
    return jsonify(res)


@app.route("/objects/<id>")
async def route_object(id):
    return await send_from_directory("objects", id)


@app.route("/logs/<uuid:id>")
async def route_logs(id):
    # Check if file exists
    if not os.path.exists(os.path.join("logs", str(id))):
        return redirect("/404")

    return await render_template("log.html", id=id, log="Loading...")

@app.route("/usage")
async def route_usage():
    return await render_template("usage.html")


# 404 error handler
@app.errorhandler(404)
async def page_not_found(_):
    return redirect("/404")


# Websocket
@app.websocket("/ws")
async def ws():
    while True:
        data = await websocket.receive()
        app.add_background_task(handle_data, data)


async def handle_data(data):
    json_data = json.loads(data)
    match json_data["type"]:
        case "running_jobs":
            with get_db() as db:
                rows = db.sql("""
                    SELECT * FROM jobs
                    WHERE started_at IS NOT NULL AND finished_at IS NULL
                    ORDER BY created_at DESC
                """).fetchall()
            jobs = row_factory_list("jobs", rows)
            await websocket.send(json.dumps({
                "type": "jobs",
                "data": jobs
            }, default=str))

        case "jobs":
            with get_db() as db:
                rows = db.sql("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
            jobs = row_factory_list("jobs", rows)
            await websocket.send(json.dumps({
                "type": "jobs",
                "data": jobs
            }, default=str))

        case "log":
            with open(os.path.join("logs", str(json_data["id"])), "r", encoding="utf-8") as f:
                await websocket.send(json.dumps({
                    "type": "log",
                    "data": f.read()
                }))

        case "research":
            res = await worker.research(json_data["query"])
            await websocket.send(json.dumps(res))

        case "search":
            papers = await worker.find_papers(json_data["query"], json_data["criterion"], json_data["sorted"])
            await websocket.send(json.dumps({
                "type": "search_papers",
                "data": papers
            }, default=str))

        case "generate_answers":
            data = await worker.generate_answers(json_data["query"], json_data["source_urls"])
            await websocket.send(json.dumps({
                "type": "answers",
                "data": data
            }))

        case "find_similar":
            papers = await worker.find_similar(json_data["source_url"])
            await websocket.send(json.dumps({
                "type": "similar_papers",
                "data": papers
            }, default=str))

        case "recursive_search":
            res = await worker.recursive_search(json_data["source_url"])
            await websocket.send(json.dumps(res))

        case "add_topic":
            res = await worker.add_topic(json_data["query"])
            await websocket.send(json.dumps(res))

        case "add_paper":
            res = await worker.add_paper(json_data["query"])
            await websocket.send(json.dumps(res))

        case "update_topic":
            res = await worker.update_topic(json_data["id"])
            await websocket.send(json.dumps(res))

        case "download_pdf":
            res = await worker.download_pdf(json_data["id"], json_data["url"])
            await websocket.send(json.dumps({
                "type": "pdf",
                "data": res
            }))

        case "usage":
            # Get some usage data
            costs = await worker.get_costs()
            summary = await worker.get_db_summary()
            await websocket.send(json.dumps({
                "type": "usage",
                "data": {
                    "costs": costs,
                    "summary": summary
                }
            }))

        case "add_to_favorites":
            print("Adding to favorites:", json_data["source_url"])
            await worker.add_to_favorites(json_data["source_url"])

        case "remove_from_favorites":
            print("Removing from favorites:", json_data["source_url"])
            await worker.remove_from_favorites(json_data["source_url"])

        case "get_paper_keywords":
            keywords = await worker.get_paper_keywords(json_data["source_url"])
            await websocket.send(json.dumps({
                "type": "paper_keywords",
                "data": keywords
            }))


# Shutdown trigger
async def shutdown_trigger():
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Raven - Accelerating the Process of Systematic Literature Reviews using Large Language Models")
    parser.add_argument("--reset", action="store_true", help="Reset the database", default=False)
    parser.add_argument("--bind", help="Bind the server to a specific address", default="127.0.0.1:8000")
    args = parser.parse_args()

    # Check arguments
    if args.reset:
        if input("Are you sure you want to reset the database? (y/N) ") != "y":
            print("Exiting...")
            exit(0)
        start = time.time()
        print("Resetting database...")
        waiter.reset()

    # Do the preparations
    waiter.check()

    # Start the worker thread
    print("Starting worker...")
    worker = Worker()
    worker.run()

    # Start the server
    try:
        config = Config()
        config._bind = [args.bind]
        asyncio.run(serve(app, config, shutdown_trigger=shutdown_trigger))
    except KeyboardInterrupt:
        print("Exiting...")
        worker.stop()

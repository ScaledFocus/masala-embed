-- SQL script to create 'labeler' table in Supabase
create table labeler (
    id serial primary key,
    name text not null,
    role text not null check (role in ('labeler', 'reviewer')),
    type text not null,
    created_at timestamp with time zone default now()
);
